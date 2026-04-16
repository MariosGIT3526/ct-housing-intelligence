import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from utils import cast_to_str

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="CT Housing Intelligence",
    page_icon="🏠",
    layout="wide"
)

# Get the directory where app.py lives
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load models and data ──────────────────────────────────────
@st.cache_resource
def load_models():
    model        = joblib.load(os.path.join(APP_DIR, 'models/final_model.pkl'))
    preprocessor = joblib.load(os.path.join(APP_DIR, 'models/preprocessor.pkl'))
    town_means   = joblib.load(os.path.join(APP_DIR, 'models/town_means.pkl'))
    global_mean  = joblib.load(os.path.join(APP_DIR, 'models/global_mean.pkl'))
    return model, preprocessor, town_means, global_mean

@st.cache_data
def load_data():
    town_metrics = pd.read_csv(os.path.join(APP_DIR, 'models/town_metrics.csv'))
    comps        = pd.read_csv(os.path.join(APP_DIR, 'models/comps_data.csv'))
    return town_metrics, comps

model, preprocessor, town_means, global_mean = load_models()
town_metrics, comps_data                     = load_data()


# ── Header ────────────────────────────────────────────────────
st.title("🏠 Connecticut Housing Intelligence")
st.markdown(
    "Predict sale prices, explore investment opportunities, "
    "and find comparable sales across Connecticut."
)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "💰 Price Predictor",
    "📈 Investment Score",
    "🔍 Comparable Sales"
])


# ═════════════════════════════════════════════════════════════
# TAB 1 — Price Predictor
# ═════════════════════════════════════════════════════════════
with tab1:
    st.header("Predict Sale Price")
    st.markdown("Enter property details to get an estimated sale price.")

    col1, col2, col3 = st.columns(3)

    with col1:
        town = st.selectbox(
            "Town",
            sorted(town_means.index.tolist())
        )
        property_type = st.selectbox("Property Type", [
            "Residential", "Single Family", "Condo",
            "Two Family", "Three Family", "Four Family"
        ])
        residential_type = st.selectbox("Residential Type", [
            "Single Family", "Condo", "Two Family",
            "Three Family", "Four Family", "Unknown"
        ])

    with col2:
        assessed_value = st.number_input(
            "Assessed Value ($)",
            min_value=0, max_value=2000000,
            value=250000, step=5000
        )
        sqft = st.number_input(
            "Square Footage (0 if unknown)",
            min_value=0, max_value=10000,
            value=0, step=100
        )
        year_built = st.number_input(
            "Year Built (0 if unknown)",
            min_value=0, max_value=2024,
            value=1980
        )

    with col3:
        beds = st.number_input(
            "Bedrooms (0 if unknown)",
            min_value=0, max_value=10,
            value=0
        )
        curr_list_year = st.number_input(
            "Listing Year",
            min_value=2001, max_value=2026,
            value=2024
        )
        curr_month = st.slider("Month of Sale", 1, 12, 6)

    if st.button("🔮 Predict Price", use_container_width=True):

        # Get town encoded value
        town_enc = town_means.get(town, global_mean)

        # Get town YoY growth
        town_yoy = (
            town_metrics[town_metrics['Town'] == town]['Avg_YoY_Growth'].values[0]
            if town in town_metrics['Town'].values else 0
        )

        # Build input row
        input_df = pd.DataFrame([{
            'Assessed Value':        assessed_value,
            'Property Type':         property_type,
            'Residential Type':      residential_type,
            'Sqft':                  sqft if sqft > 0 else np.nan,
            'Beds':                  beds if beds > 0 else np.nan,
            'Garage':                np.nan,
            'Year Built':            year_built if year_built > 0 else np.nan,
            'Days_To_Sell':          180,
            'curr_list_year':        curr_list_year,
            'prev_list_year':        curr_list_year,
            'curr_month_recorded':   curr_month,
            'Flip_Candidate':        0,
            'Is_Older_Home':         1 if year_built > 0 and year_built < 1980 else 0,
            'List_Year_Num':         curr_list_year,
            'Home_Age':              curr_list_year - year_built if year_built > 0 else np.nan,
            'Luxury_Home':           1 if beds >= 5 and sqft >= 4000 else 0,
            'town_encoded':          town_enc,
            'town_yoy_growth':       town_yoy,
        }])

        input_df = input_df.replace([np.inf, -np.inf], np.nan)

        # Predict
        processed  = preprocessor.transform(input_df)
        pred_log   = model.predict(processed)
        predicted  = np.expm1(pred_log)[0]

        # Confidence interval
        lower = max(0, predicted - 133000)
        upper = predicted + 133000

        # Results
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Sale Price", f"${predicted:,.0f}")
        c2.metric("Lower Bound",          f"${lower:,.0f}")
        c3.metric("Upper Bound",          f"${upper:,.0f}")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted,
            number={'prefix': "$", 'valueformat': ",.0f"},
            gauge={
                'axis':  {'range': [0, 2000000]},
                'bar':   {'color': "steelblue"},
                'steps': [
                    {'range': [0, 200000],       'color': "#e8f4f8"},
                    {'range': [200000, 500000],   'color': "#b8d4e8"},
                    {'range': [500000, 1000000],  'color': "#7aafcf"},
                    {'range': [1000000, 2000000], 'color': "#3d8ab5"},
                ],
            },
            title={'text': f"Predicted Price — {town}"}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Town market context
        town_row = town_metrics[town_metrics['Town'] == town]
        if not town_row.empty:
            st.markdown("### 📊 Market Context")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Investment Score", f"{town_row['Investment_Score'].values[0]:.3f}")
            c2.metric("Avg YoY Growth",   f"{town_row['Avg_YoY_Growth'].values[0]:.1f}%")
            c3.metric("Recent Momentum",  f"{town_row['Recent_Momentum'].values[0]:.1f}%")
            c4.metric("Avg Days to Sell", f"{town_row['Avg_Days_To_Sell'].values[0]:.0f} days")


# ═════════════════════════════════════════════════════════════
# TAB 2 — Investment Score
# ═════════════════════════════════════════════════════════════
with tab2:
    st.header("Town Investment Rankings")
    st.markdown(
        "Composite investment score based on YoY price growth, "
        "recent momentum, flip rate, and market liquidity."
    )

    col1, col2 = st.columns([2, 1])

    with col2:
        min_transactions = st.slider(
            "Min transactions filter", 500, 5000, 500, step=500
        )
        top_n    = st.slider("Show top N towns", 10, 50, 20)
        color_by = st.selectbox("Color by", [
            "Investment_Score", "Avg_YoY_Growth",
            "Recent_Momentum",  "Flip_Rate"
        ])

    filtered = (
        town_metrics[town_metrics['Transaction_Count'] >= min_transactions]
        .sort_values('Investment_Score', ascending=False)
        .head(top_n)
    )

    with col1:
        fig = px.bar(
            filtered,
            x='Investment_Score',
            y='Town',
            orientation='h',
            color=color_by,
            color_continuous_scale='viridis',
            title=f'Top {top_n} Towns by Investment Score',
            labels={'Investment_Score': 'Investment Score (0-1)'}
        )
        fig.update_layout(
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Full rankings table
    st.markdown("### Full Rankings Table")
    display_cols = [
        'Rank', 'Town', 'Investment_Score', 'Avg_YoY_Growth',
        'Recent_Momentum', 'Flip_Rate', 'Avg_Days_To_Sell',
        'Median_Price', 'Transaction_Count'
    ]
    st.dataframe(
        town_metrics[display_cols]
        .sort_values('Rank')
        .style.format({
            'Investment_Score': '{:.3f}',
            'Avg_YoY_Growth':   '{:.1f}%',
            'Recent_Momentum':  '{:.1f}%',
            'Flip_Rate':        '{:.3f}',
            'Avg_Days_To_Sell': '{:.0f}',
            'Median_Price':     '${:,.0f}',
        }),
        use_container_width=True,
        height=400
    )


# ═════════════════════════════════════════════════════════════
# TAB 3 — Comparable Sales
# ═════════════════════════════════════════════════════════════
with tab3:
    st.header("Find Comparable Sales")
    st.markdown(
        "Search recent sales in a town to find properties "
        "similar to yours."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        comp_town = st.selectbox(
            "Town",
            sorted(comps_data['Town'].unique()),
            key='comp_town'
        )
    with col2:
        comp_type = st.selectbox(
            "Property Type",
            ['All'] + sorted(
                comps_data['Residential Type'].dropna().unique().tolist()
            )
        )
    with col3:
        comp_years = st.slider(
            "Year Range", 2015, 2024, (2018, 2024)
        )

    price_range = st.slider(
        "Sale Price Range ($)",
        0, 2000000, (0, 2000000),
        step=10000
    )

    # Filter
    mask = (
        (comps_data['Town'] == comp_town) &
        (comps_data['prev_list_year'] >= comp_years[0]) &
        (comps_data['prev_list_year'] <= comp_years[1]) &
        (comps_data['Sale Amount']    >= price_range[0]) &
        (comps_data['Sale Amount']    <= price_range[1])
    )
    if comp_type != 'All':
        mask &= comps_data['Residential Type'] == comp_type

    results = comps_data[mask].sort_values('Sale Amount', ascending=False)

    st.markdown(f"**{len(results):,} comparable sales found**")

    if len(results) > 0:
        # Summary stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median Price",  f"${results['Sale Amount'].median():,.0f}")
        c2.metric("Average Price", f"${results['Sale Amount'].mean():,.0f}")
        c3.metric("Lowest",        f"${results['Sale Amount'].min():,.0f}")
        c4.metric("Highest",       f"${results['Sale Amount'].max():,.0f}")

        # Price distribution
        fig = px.histogram(
            results,
            x='Sale Amount',
            nbins=40,
            title=f'Sale Price Distribution — {comp_town}',
            labels={'Sale Amount': 'Sale Price ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Price over time
        yearly = (
            results.groupby('prev_list_year')['Sale Amount']
            .median()
            .reset_index()
        )
        fig2 = px.line(
            yearly,
            x='prev_list_year',
            y='Sale Amount',
            markers=True,
            title=f'Median Sale Price Over Time — {comp_town}',
            labels={
                'prev_list_year': 'Year',
                'Sale Amount':    'Median Price ($)'
            }
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Raw table
        st.dataframe(
            results[[
                'Address', 'Sale Amount', 'Assessed Value',
                'Residential Type', 'Sqft', 'Beds',
                'Year Built', 'prev_list_year'
            ]]
            .rename(columns={'prev_list_year': 'Year'})
            .style.format({
                'Sale Amount':    '${:,.0f}',
                'Assessed Value': '${:,.0f}',
                'Sqft':           '{:,.0f}',
            }),
            use_container_width=True,
            height=400
        )
    else:
        st.warning(
            "No comparable sales found. "
            "Try adjusting the year range or price range."
        )


# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Built with XGBoost + Bayesian Optimization | "
    "1.1M CT transactions (2001–2023) | "
    "Model RMSE: $115,231"
)