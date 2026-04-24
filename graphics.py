import os
import pandas as pd
import plotly.express as px

os.makedirs("output", exist_ok=True)
with open('Stats_omp.txt','w', encoding='utf-8') as f:
    f.write("N;threads;time_ms;ops\n200;1;104;16000000\n200;2;102;16000000\n200;4;104;16000000\n200;8;112;16000000\n400;1;841;128000000\n400;2;858;128000000\n400;4;815;128000000\n400;8;705;128000000\n800;1;5149;1024000000\n800;2;5161;1024000000\n800;4;5240;1024000000\n800;8;5171;1024000000\n1200;1;20088;3456000000\n1200;2;19922;3456000000\n1200;4;19852;3456000000\n1200;8;19853;3456000000\n1600;1;47936;8192000000\n1600;2;48074;8192000000\n1600;4;47934;8192000000\n1600;8;49950;8192000000\n2000;1;107912;16000000000\n2000;2;100644;16000000000\n2000;4;106015;16000000000\n2000;8;113060;16000000000")


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