import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Decision Support Tool", layout="wide")

# --- Load raw dataset and run optimization automatically ---
data_path = "./dataset.csv"
df_raw = pd.read_csv(data_path)

# --- Preprocess objectives ---
objectives = ['out:EUI', 'out:CO2', 'out:Cost', 'out:Area', 'out:dgp', 'out:udi']
df = df_raw.copy()

# Parse 'out:dgp' assuming format 'val1;val2;val3;val4'
def parse_dgp(val):
    try:
        return float(str(val).split(';')[-1])
    except:
        return np.nan

df['out:dgp_parsed'] = df['out:dgp'].apply(parse_dgp)
df = df.dropna(subset=objectives + ['out:dgp_parsed'])

F = df[['out:EUI', 'out:CO2', 'out:Cost', 'out:Area', 'out:udi', 'out:dgp_parsed']].copy()
F['out:Area'] = 1 - (F['out:Area'] - F['out:Area'].min()) / (F['out:Area'].max() - F['out:Area'].min() + 1e-6)
F['out:udi'] = 1 - (F['out:udi'] - F['out:udi'].min()) / (F['out:udi'].max() - F['out:udi'].min() + 1e-6)
F['out:dgp_parsed'] = (F['out:dgp_parsed'] - F['out:dgp_parsed'].min()) / (F['out:dgp_parsed'].max() - F['out:dgp_parsed'].min() + 1e-6)
for col in ['out:EUI', 'out:CO2', 'out:Cost']:
    F[col] = (F[col] - F[col].min()) / (F[col].max() - F[col].min() + 1e-6)

F_matrix = F.to_numpy()

# --- Optimization ---
class EvaluatedProblem(Problem):
    def __init__(self, F):
        super().__init__(n_var=1, n_obj=F.shape[1], n_constr=0, xl=0, xu=1)
        self.F = F

    def _evaluate(self, x, out, *args, **kwargs):
        idx = np.clip((x[:, 0] * (self.F.shape[0] - 1)).astype(int), 0, self.F.shape[0] - 1)
        out["F"] = self.F[idx]

problem = EvaluatedProblem(F_matrix)
algorithm = NSGA2(pop_size=100)
termination = get_termination("n_gen", 100)
res = minimize(problem, algorithm, termination, seed=1, save_history=False, verbose=False)

indices = [int(ind.X[0] * (F_matrix.shape[0] - 1)) for ind in res.pop]
df_pareto = df.iloc[indices].drop_duplicates(subset='Run').copy()
df_pareto['Front'] = 1

nds = NonDominatedSorting()
fronts = nds.do(F_matrix, only_non_dominated_front=False)
for i, front in enumerate(fronts[1:6], start=2):
    front_df = df.iloc[front].drop_duplicates().copy()
    front_df['Front'] = i
    df_pareto = pd.concat([df_pareto, front_df], ignore_index=True)

# --- Normalize for value function ---
normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
df_pareto['norm_EUI'] = normalize(df_pareto['out:EUI'])
df_pareto['norm_CO2'] = normalize(df_pareto['out:CO2'])
df_pareto['norm_Cost'] = normalize(df_pareto['out:Cost'])
df_pareto['norm_Area'] = 1 - normalize(df_pareto['out:Area'])
df_pareto['norm_udi'] = 1 - normalize(df_pareto['out:udi'])
df_pareto['norm_dgp'] = normalize(df_pareto['out:dgp_parsed'])

# --- Weights from sidebar ---
st.sidebar.header("🔧 Set AHP Weights")
w_eui = st.sidebar.slider("EUI", 0.0, 1.0, 0.2, 0.01)
w_co2 = st.sidebar.slider("CO2", 0.0, 1.0, 0.2, 0.01)
w_cost = st.sidebar.slider("Cost", 0.0, 1.0, 0.2, 0.01)
w_area = st.sidebar.slider("Area", 0.0, 1.0, 0.2, 0.01)
w_udi = st.sidebar.slider("UDI", 0.0, 1.0, 0.1, 0.01)
w_dgp = st.sidebar.slider("DGP", 0.0, 1.0, 0.1, 0.01)

total = w_eui + w_co2 + w_cost + w_area + w_udi + w_dgp
if total > 0:
    w_eui /= total
    w_co2 /= total
    w_cost /= total
    w_area /= total
    w_udi /= total
    w_dgp /= total

# --- Value function ---
df_pareto['Value_Function'] = (
    w_eui * df_pareto['norm_EUI'] +
    w_co2 * df_pareto['norm_CO2'] +
    w_cost * df_pareto['norm_Cost'] +
    w_area * df_pareto['norm_Area'] +
    w_udi * df_pareto['norm_udi'] +
    w_dgp * df_pareto['norm_dgp']
)
df_pareto['AHP_Score'] = df_pareto['Value_Function']

# --- Filter by Levels ---
st.sidebar.header("🏢 Filter by Number of Levels")
selected_levels = []
for level in [2, 3, 4]:
    if st.sidebar.checkbox(f"{level} Levels", value=True):
        selected_levels.append(level)

# --- Fronts selection ---
st.sidebar.header("📂 Select Fronts")
selected_fronts = [i for i in range(1, 7) if st.sidebar.checkbox(f"Front {i}", value=(i == 1))]
df_ranked = df_pareto[df_pareto['Front'].isin(selected_fronts)]
if 'in:levels' in df_ranked.columns:
    df_ranked = df_ranked[df_ranked['in:levels'].isin(selected_levels)].sort_values("AHP_Score").reset_index(drop=True)
df_ranked['Rank'] = df_ranked.index + 1
df_ranked['Run'] = df_ranked['Run'] if 'Run' in df_ranked.columns else df_ranked.index

# --- Export ranked results ---
st.sidebar.header("⬇️ Export Ranked Alternatives")
num_to_export = st.sidebar.number_input("Number of top alternatives to export", min_value=1, max_value=len(df_ranked), value=10, step=1)
export_df = df_ranked.head(num_to_export)

csv = export_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download CSV",
    data=csv,
    file_name='ranked_designs.csv',
    mime='text/csv'
)

# --- Display results ---
st.title("📊 Decision Support Tool")
st.subheader("🏅 Ranked Design Alternatives")
st.dataframe(df_ranked[['Rank', 'Run', 'out:EUI', 'out:CO2', 'out:Cost', 'out:Area', 'out:udi', 'out:dgp_parsed', 'AHP_Score', 'Front']])

# --- Image Viewer ---
st.subheader("🖼️ Design Images")
image_mode = st.radio("Select image type to view:", ['DGP', 'UDI', 'Floor'], horizontal=True)

image_dirs = {
    'DGP': "DGP",
    'UDI': "UDI/UDI",
    'floor': "floor",
}
img_column_map = {
    'DGP': 'img:DGP',
    'UDI': 'img:udi',
    'Floor': 'img:floor'
}
cols = st.columns(3)
for i, row in df_ranked.iterrows():
    img_name = row.get(img_column_map[image_mode])
    if isinstance(img_name, str):
        img_path = os.path.join(image_dirs[image_mode], img_name)
        with cols[i % 3]:
            st.markdown(f"**Rank {row['Rank']} – Run {row['Run']}**")
            if os.path.exists(img_path):
                try:
                    st.image(img_path, caption=f"{image_mode} | Score: {row['AHP_Score']:.3f}", use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Could not display image: {img_name} — {e}")
            else:
                st.warning(f"Image not found: {img_name}")

# --- Sensitivity Analysis ---
st.sidebar.header("📈 Top Sensitive Inputs")
input_cols = [col for col in df_raw.columns if col.startswith("in:")]
df_inputs = df_ranked[input_cols].copy()
df_inputs = pd.get_dummies(df_inputs)
df_inputs = df_inputs.loc[:, df_inputs.nunique() > 1]  # avoid NaN in correlation
sa_target = df_ranked['Value_Function']
sensitivity_series = df_inputs.apply(lambda x: x.corr(sa_target)).abs()

# Group dummy variable sensitivities by category
sensitivity_df = sensitivity_series.reset_index()
sensitivity_df.columns = ['feature', 'corr']
sensitivity_df['category'] = sensitivity_df['feature'].str.extract(r'(.*?)(?:_|$)')
grouped_corr = sensitivity_df.groupby('category')['corr'].max().sort_values(ascending=False).head(7)

df_inputs = df_inputs.loc[:, df_inputs.nunique() > 1]  # avoid NaN in correlation


fig_sens = px.bar(
    x=grouped_corr.values,
    y=grouped_corr.index,
    orientation='h',
    labels={'x': 'Max Correlation', 'y': 'Input Category'}
)
fig_sens.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
st.sidebar.plotly_chart(fig_sens, use_container_width=True)

# --- NSGA-II Fronts Plot ---
# st.subheader("📌 Optimization Fronts (EUI vs Area)")
# plt.figure(figsize=(12, 8))
# plt.scatter(df_raw['out:Area'], df_raw['out:EUI'], color='lightgray', alpha=0.5, label='All Designs')
# colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
# for i, color in zip(range(1, 7), colors):
#     front_df = df_pareto[df_pareto['Front'] == i]
#     if not front_df.empty and i in selected_fronts:
#         plt.plot(front_df['out:Area'], front_df['out:EUI'], color=color, label=f"Front {i}")
# plt.xlabel("Area")
# plt.ylabel("EUI")
# plt.legend()
# plt.grid(True)
# st.pyplot(plt.gcf())

# --- Parallel Coordinates Plot ---
# st.subheader("📈 Parallel Coordinates Plot")
# pcp_cols = [
#     'in:levels', 'in:shift', 'in:levelfraction', 'in:floorthickness', 'in:fractionextension',
#     'in:wallrvalue', 'in:roofrvalue', 'in:floortype',
#     'in:hvacheating_type', 'in:hvachotwater', 'in:hvacecon', 'in:hvacheatrecovery', 'in:buildinglpd', 'in:hvaccold',
#     'out:EUI', 'out:CO2', 'out:Cost', 'out:Area', 'out:dgp_parsed'
# ]
# present_cols = [col for col in pcp_cols if col in df_ranked.columns]

# highlighted = df_ranked.nsmallest(1, 'AHP_Score').copy().copy()
# others = df_ranked[~df_ranked.index.isin(highlighted.index)].copy().copy()
# highlighted['color_val'] = 1
# others['color_val'] = 0
# combined = pd.concat([highlighted, others])

# fig = px.parallel_coordinates(
#     combined,
#     dimensions=present_cols,
#     color='color_val',
#     color_continuous_scale=[[0, 'black'], [1, 'red']],
#     range_color=[0, 1]
# )
# fig.update_layout(width=2000, height=600)
# st.plotly_chart(fig)
