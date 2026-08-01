import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# CONFIG 
st.set_page_config(page_title="Career Dashboard", layout="wide")

# CUSTOM CSS 
st.markdown("""
<style>
.kpi-box {
    width: 145px;
    height: 120px;
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-weight: bold;
}
.kpi1 { background-color: #3b6e7a; }   /* Muted Teal Blue */
.kpi2 { background-color: #e6a23c; }   /* Soft Orange */
.kpi3 { background-color: #d65a63; }   /* Soft Red */
.kpi4 { background-color: #5b8def; }   /* Clean Blue */
.kpi5 { background-color: #6c9a8b; }   /* Sage Green */
            
.card {
    width: 145px;
    height: 120px;
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-weight: bold;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
.card-1 { background-color: #3b6e7a; }   /* Muted Teal Blue */
.card-2 { background-color: #e6a23c; }   /* Soft Orange */
.card-3 { background-color: #d65a63; }   /* Soft Red */
.card-4 { background-color: #5b8def; }   /* Clean Blue */
.card-5 { background-color: #6c9a8b; }   /* Sage Green */

.card-title {
    font-size: 14px;
}
.card-value {
    font-size: 26px;
    margin-top: 5px;
}

.kpi-card {
    height: 110px;
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-weight: 600;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;  
}
.kpi-title {
    font-size: 14px;
    opacity: 0.9;
}
.kpi-value {
    font-size: 28px;
    margin-top: 5px;
}
.red {background: #e74c3c;}
.orange {background: #f39c12;}
.blue {background: #3498db;}
.green {background: #2ecc71;}

</style>
""", unsafe_allow_html=True)

# LOAD DATA 
@st.cache_data
def load_data():
    return pd.read_csv("Palo Alto Networks.csv")

df = load_data()

# KPI CALCULATION 
df["PromotionGapRatio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
df["RoleStagnationIndex"] = df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)
df["TrainingNeedIndicator"] = 1 / (df["TrainingTimesLastYear"] + 1)
df["ManagerStabilityImpact"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)

df["RetentionOpportunityIndex"] = (
    df["PromotionGapRatio"] * 0.4 +
    df["RoleStagnationIndex"] * 0.3 +
    df["TrainingNeedIndicator"] * 0.2 +
    (1 - df["ManagerStabilityImpact"]) * 0.1
)

# CLUSTERING 
features = [
    "Age", "MonthlyIncome", "JobLevel",
    "JobSatisfaction", "EnvironmentSatisfaction",
    "WorkLifeBalance", "PromotionGapRatio",
    "RoleStagnationIndex", "TrainingNeedIndicator",
    "ManagerStabilityImpact"
]

scaler = StandardScaler()
scaled = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=5, random_state=42)
df["CareerCluster"] = kmeans.fit_predict(scaled)

# SIDEBAR 
st.sidebar.title("Filters")

dept = st.sidebar.multiselect("Department", df["Department"].unique(), default=df["Department"].unique())
role = st.sidebar.multiselect("Job Role", df["JobRole"].unique(), default=df["JobRole"].unique())
def get_career_stage(exp):
    if exp <= 3:
        return "Early Career"
    elif exp <= 10:
        return "Mid Career"
    else:
        return "Senior Leadership"
df["CareerStage"] = df["TotalWorkingYears"].apply(get_career_stage)
career_stage = st.sidebar.multiselect("Career Stage",df["CareerStage"].unique(),default=df["CareerStage"].unique())
gap_threshold = st.sidebar.slider("Promotion Gap Threshold", 0.0, 1.0, 0.5)

filtered_df = df[
    (df["Department"].isin(dept)) &
    (df["JobRole"].isin(role)) &
    (df["CareerStage"].isin(career_stage))
]

# HEADER
st.title("Career Analytics Dashboard")

# KPI BOXES (5 KPIs)
k1, k2, k3, k4, k5 = st.columns(5)

# 1. Total Employees
k1.markdown(f'<div class="kpi-box kpi1">Total Employees<br>{len(filtered_df)}</div>', unsafe_allow_html=True)

# 2. Avg Promotion Gap
avg_gap = round(filtered_df["PromotionGapRatio"].mean(), 2)
k2.markdown(f'<div class="kpi-box kpi2">Avg Promotion Gap<br>{avg_gap}</div>', unsafe_allow_html=True)

# 3. High Risk Count
high_risk = len(filtered_df[filtered_df["PromotionGapRatio"] > gap_threshold])
k3.markdown(f'<div class="kpi-box kpi3">High Risk Employees<br>{high_risk}</div>', unsafe_allow_html=True)

# 4. Retention Opportunity Count
retention_count = len(filtered_df[filtered_df["RetentionOpportunityIndex"] > 0.6])
k4.markdown(f'<div class="kpi-box kpi4">Retention Opportunity<br>{retention_count}</div>', unsafe_allow_html=True)

# 5. Avg Training Score
filtered_df["Training_Intensity_Score"] = (
    filtered_df["TrainingTimesLastYear"] /
    filtered_df["YearsAtCompany"].replace(0, 1)
)
avg_training = round(filtered_df["Training_Intensity_Score"].mean(), 2)
k5.markdown(f'<div class="kpi-box kpi5">Avg Training Score<br>{avg_training}</div>', unsafe_allow_html=True)

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "Career Clustering",
    "Promotion Gap",
    "Retention Panel",
    "Manager Insights"
])

#  TAB 1
with tab1:
    st.subheader("Career Path Clustering")
    st.markdown("#### Cluster Distribution")
    fig1 = px.histogram(filtered_df, x="CareerCluster", color="CareerCluster")
    st.plotly_chart(fig1, use_container_width=True)  

    # Base dataframe (clusters)
    st.markdown("#### Career Pattern Summaries")
    summary_df = filtered_df[["CareerCluster"]].drop_duplicates().sort_values("CareerCluster")
    filtered_df["Education"] = filtered_df["Education"].replace({
    1: "Below College",
    2: "College",
    3: "Bachelor",
    4: "Master",
    5: "Doctor"
    })

#  Job Role
    jobrole = filtered_df.groupby("CareerCluster")["JobRole"] \
        .agg(lambda x: x.value_counts().index[0])

#  Education
    education = filtered_df.groupby("CareerCluster")["Education"] \
        .agg(lambda x: x.value_counts().index[0])

#  Department
    department = filtered_df.groupby("CareerCluster")["Department"] \
        .agg(lambda x: x.value_counts().index[0])

#  Income
    income = filtered_df.groupby("CareerCluster")["MonthlyIncome"].mean()

#  Experience
    experience = filtered_df.groupby("CareerCluster")["TotalWorkingYears"].mean()

# Merge all into one table
    summary_df["Top Job Role"] = summary_df["CareerCluster"].map(jobrole)
    summary_df["Top Education"] = summary_df["CareerCluster"].map(education)
    summary_df["Top Department"] = summary_df["CareerCluster"].map(department)
    summary_df["Avg Income"] = summary_df["CareerCluster"].map(lambda x: int(income[x]))
    summary_df["Avg Experience"] = summary_df["CareerCluster"].map(lambda x: round(experience[x],1))
    st.dataframe(summary_df)

    # CLUSTER EXPLORER 
    st.subheader("Cluster Explorer")
    if "CareerCluster" in filtered_df.columns:
        if not filtered_df.empty:
            cluster_select = st.selectbox(
                "Select Cluster",
                sorted(filtered_df["CareerCluster"].dropna().unique())
            )
            cluster_df = filtered_df[filtered_df["CareerCluster"] == cluster_select]

            col1, col2, col3 = st.columns(3)

            col1.markdown(f"""
            <div class="card card-1">
                <div class="card-title">Employees</div>
                <div class="card-value">{len(cluster_df)}</div>
            </div>
            """, unsafe_allow_html=True)

            col2.markdown(f"""
            <div class="card card-2">
            <div class="card-title">Avg Income</div>
                <div class="card-value">{int(cluster_df["MonthlyIncome"].mean())}</div>
            </div>
            """, unsafe_allow_html=True)

            col3.markdown(f"""
            <div class="card card-3">
                <div class="card-title">Avg Experience</div>
                <div class="card-value">{round(cluster_df["TotalWorkingYears"].mean(),1)}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Summary Stats")
            st.dataframe(cluster_df.describe())

            st.markdown("#### Employee Data")
            # here to add
            cols = [
                "Age",
                "Department",
                "JobRole",
                "MonthlyIncome",
                "YearsAtCompany",
                "YearsSinceLastPromotion",
                "PromotionGapRatio",
                "RoleStagnationIndex",
                "TrainingNeedIndicator",
                "ManagerStabilityImpact",
                "RetentionOpportunityIndex",
                "CareerCluster"
            ]
            cluster_df_display = cluster_df[cols].reset_index(drop=True)
            cluster_df_display.insert(0, "Sr No", range(1, len(cluster_df_display)+1))

            st.dataframe(cluster_df_display, hide_index=True)
            
        else:
            st.warning("No data available for selected filters.")
    else:
        st.error("CareerCluster column not found.")

# TAB 2 
with tab2:
    st.subheader("Promotion Gap Monitor")

    role_order = (
    filtered_df.groupby("JobRole")["PromotionGapRatio"]
    .median()
    .sort_values(ascending=False)
    .index
    )

    fig2 = px.box(
        filtered_df,
        x="JobRole",
        y="PromotionGapRatio",
        category_orders={"JobRole": role_order},
        points="outliers",
        title="Promotion Gap Distribution by Job Role"
    )

    fig2.add_hline(
        y=gap_threshold,
        line_dash="dash",
        annotation_text="Risk Threshold",
        annotation_position="top left"
    )

    fig2.update_layout(
        xaxis_tickangle=-30
    )

    st.plotly_chart(fig2, use_container_width=True)

    #  Risk Classification Function
    def classify_risk(val):
        if val > 0.6:
            return "High Risk"
        elif val > 0.3:
            return "Medium Risk"
        else:
            return "Low Risk"

    # RiskLevel column create 
    filtered_df = filtered_df.copy() 
    filtered_df["RiskLevel"] = filtered_df["PromotionGapRatio"].apply(classify_risk)

    # Scatter Plot 
    import plotly.express as px

    fig = px.scatter(
        filtered_df,
        x="YearsAtCompany",
        y="PromotionGapRatio",
        color="RiskLevel",
        title="Employee Promotion Risk Distribution",
        hover_data=["JobRole", "Department"],
         color_discrete_map={
        "High Risk": "red",
        "Medium Risk": "orange",
        "Low Risk": "yellow"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df["EmpID"] = filtered_df.index + 1

    # High-gap employee identification
    st.markdown("####  High Promotion Gap Employees")

    high_gap_df = filtered_df[filtered_df["PromotionGapRatio"] > gap_threshold]
    high_gap_df = high_gap_df.sort_values(by="PromotionGapRatio", ascending=False)
    st.dataframe(
        high_gap_df[[
            "EmpID",
            "JobRole",
            "YearsAtCompany",
            "YearsSinceLastPromotion",
            "PromotionGapRatio"
        ]],
        hide_index=True
    )
# TAB 3 
with tab3:
    st.subheader("Retention Opportunity Panel")
    fig3 = px.scatter(
        filtered_df,
        x="PromotionGapRatio",
        y="RetentionOpportunityIndex",
        color="RetentionOpportunityIndex",
        size="YearsAtCompany",
        hover_data=["JobRole", "Department"],
        title=" Retention Opportunity Distribution"
    )
    st.plotly_chart(fig3, use_container_width=True)

    #  Suggested Action Logic
    def suggest_action(row):
        if row["PromotionGapRatio"] > 0.6:
            return "Promotion Review"
        elif row["TrainingNeedIndicator"] > 0.5:
            return "Training Required"
        elif row["RoleStagnationIndex"] > 0.5:
            return "Job Rotation"
        else:
            return "Monitor"

    filtered_df = filtered_df.copy()
    filtered_df["SuggestedAction"] = filtered_df.apply(suggest_action, axis=1)

    st.markdown("####  Employees Needing Career Intervention")
    high_retention_df = filtered_df[
        filtered_df["RetentionOpportunityIndex"] > 0.6
    ].sort_values(by="RetentionOpportunityIndex", ascending=False)

    intervention_count = len(high_retention_df)
    high_gap_count = len(filtered_df[filtered_df["PromotionGapRatio"] > gap_threshold])
    avg_risk = round(filtered_df["RetentionOpportunityIndex"].mean(), 2)
    max_risk = round(filtered_df["RetentionOpportunityIndex"].max(), 2)
    def kpi_card(title, value, color):
        return f"""
            <div class="kpi-card {color}">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            </div>
    """

    high_retention_df = filtered_df[filtered_df["RetentionOpportunityIndex"] > 0.6]
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(kpi_card(" Intervention Needed", intervention_count, "red"), unsafe_allow_html=True)
    col2.markdown(kpi_card(" High Promotion Gap", high_gap_count, "orange"), unsafe_allow_html=True)
    col3.markdown(kpi_card(" Avg Retention Risk", avg_risk, "blue"), unsafe_allow_html=True)
    col4.markdown(kpi_card(" Max Risk Score", max_risk, "green"), unsafe_allow_html=True)
  
    st.dataframe(
        high_retention_df[[
            "EmpID",
            "JobRole",
            "Department",
            "PromotionGapRatio",
            "RetentionOpportunityIndex"
        ]].sort_values(by="RetentionOpportunityIndex", ascending=False),
        hide_index=True
    )

    st.markdown("####  Suggested Actions")

    st.dataframe(
        high_retention_df[[
            "EmpID",
            "JobRole",
            "Department",
            "RetentionOpportunityIndex",
            "SuggestedAction"
        ]].sort_values(by="RetentionOpportunityIndex", ascending=False),
        hide_index=True
    )
    
# TAB 4 
with tab4:
    st.subheader("Managerial Insight Dashboard")

    # Manager Tenure vs Career Growth

    fig4 = px.scatter(
        filtered_df,
        x="YearsWithCurrManager",
        y="PromotionGapRatio",
        color="Department",
        size="YearsAtCompany",
        hover_data=["JobRole"],
        title=("Manager Tenure vs Promotion Gap Ratio")
    )

    fig4.update_layout(
        xaxis_title="Years with Current Manager",
        yaxis_title="Promotion Gap Ratio"
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Team-Level Stagnation Signals
    st.markdown("####  Team-Level Stagnation Signals")

    team_stagnation = (
        filtered_df
        .groupby("Department")["PromotionGapRatio"]
        .mean()
        .reset_index()
        .sort_values(by="PromotionGapRatio", ascending=False)
    )

    fig_team = px.bar(
        team_stagnation,
        x="Department",
        y="PromotionGapRatio",
        text_auto=True
    )

    fig_team.update_layout(
        yaxis_title="Avg Promotion Gap Ratio"
    )

    st.plotly_chart(fig_team, use_container_width=True)

    st.markdown("####  Department-wise Risk Table")

    st.dataframe(
        team_stagnation,
        hide_index=True
    )