import os
import pandas as pd
import plotly.express as px

os.makedirs("output", exist_ok=True)

df = pd.read_csv('Stats_omp.txt', sep=';')

sizes = sorted(df["N"].unique())

fig2 = px.line(
    df,
    x="N",
    y="time_ms",
    color="threads",
    markers=True,
    title="Зависимость времени выполнения от размера матрицы",
    labels={
        "N": "Размер матрицы N",
        "time_ms": "Время выполнения, мс",
        "threads": "Количество потоков"
    }
)

fig2.update_xaxes(
    title_text="Размер матрицы N",
    tickmode="array",
    tickvals=sizes,
    ticktext=[str(n) for n in sizes]
)
fig2.update_yaxes(title_text="Время выполнения, мс")

fig2.write_image("output/time_vs_size.png", scale=2)