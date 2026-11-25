#Ferramentas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller,kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
arquivo="/home/gabriel/Área de trabalho/Ultimo Trabalho de series/US_inflation_rates.csv"
df=pd.read_csv(arquivo,parse_dates=["date"]).set_index("date")
y=df["value"].asfreq("MS")

df.plot(figsize=(12,4))
plt.title("Taxa de inflação EUA")
plt.xlabel("date")
plt.ylabel("value")
plt.show()


df.info()
df.describe()
print(f"Linhas:{df.shape[0]}")

#STL para separar tendencia,sazonalidade e residuo

stl=STL(y,period=12,robust=True).fit() #robust = Iginorar outlouer o fit é o que executa o calculo
stl.plot()
plt.show()

#ADF e Kpss antes do diff
adf=adfuller(y)
print(f"ADF p-valor: {adf[1]:.4f}") #ADF p-valor: 0.9991

kps=kpss(y,regression="c",nlags="auto")
print(f"KPSS p-valor: {kps[1]:.4f}\n") #KPSS p-valor: 0.0100

#Série nao estacionaria
# FAC da série original (decai devagar = não estacionária)
plot_acf(y, lags=36)
plt.show()

#1 diferenciação
y_dif=y.diff().dropna()

plt.figure(figsize=(12,4))
plt.plot(
    y_dif.index,
    y_dif.values,
    color="darkorange",
    label="Primeira diferença"
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Taxa de Inflação EUA - Após 1ª Diferenciação")
plt.xlabel("date")
plt.ylabel("Variação")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#ADF e Kpss apos do diff
adf_diff=adfuller(y_dif)
print(f"ADF p-valor: {adf_diff[1]:.4f}") #ADF_diff p-valor: 0.0061

kps_diff=kpss(y_dif,regression="c",nlags="auto")
print(f"KPSS p-valor: {kps_diff[1]:.4f}\n") #KPSS_diff p-valor: 0.0100

#DIFERENÇA SAZONAL (D=1, s=12) SOBRE A 1ª DIFERENÇA

y_dif_saz=y_dif.diff(12).dropna()


plt.figure(figsize=(12, 4))
plt.plot(
    y_dif_saz.index,
    y_dif_saz.values,
    color="purple",
    label="1ª Dif + Dif Sazonal (d=1, D=1, s=12)",
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Taxa de Inflação EUA - Após Diferença Regular (d=1) e Sazonal (D=1)")
plt.xlabel("date")
plt.ylabel("Variação")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Testando ADF e KPSS novamente
adf_saz = adfuller(y_dif_saz)
kps_saz = kpss(y_dif_saz, regression="c", nlags="auto")

print(f"Dif + Sazonal = ADF p-valor:  {adf_saz[1]:.4f}") #Dif + Sazonal = ADF p-valor:  0.0000
print(f"Dif + Sazonal = KPSS p-valor: {kps_saz[1]:.4f}") #Dif + Sazonal = KPSS p-valor: 0.1000

#Os dois bateu é estacionaria

serie_estacionaria=y_dif_saz

#FAC e FACP
fig, axes= plt.subplots(1, 2, figsize=(14, 4))

plot_acf(serie_estacionaria,lags=36,ax=axes[0],title="FAC (ACF) - Série Estacionária")

plot_pacf(serie_estacionaria, lags=36, ax=axes[1], title="FACP (PACF) - Série Estacionária")

plt.tight_layout()
plt.show()

#MA(1) 

#Modelo SARIMA(1,1,1)(0,1,1,12)

n_teste = 12 #O numero de meses que vou esconder do modelo
y_treino = y.iloc[:-n_teste] #corta os 12 ultimos da fila e fica so com o primeiro
y_teste = y.iloc[-n_teste:] #Fica com os 12 meses em se

# 2. Configurando e treinando o modelo SARIMA(1,1,1)(0,1,1,12)
modelo = SARIMAX(
    y_treino, #Sao os dados de estudo passado
    order=(1, 1, 1),
    seasonal_order=(0, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)

ajuste = modelo.fit(disp=False)

# 3. Exibindo o resumo estatístico (coeficientes, p-valores, AIC e BIC)
print(ajuste.summary())



# 4. Teste de Ljung-Box nos resíduos (queremos lb_pvalue > 0.05)
residuos = ajuste.resid.dropna()

print(acorr_ljungbox(residuos, lags=[12, 24], return_df=True))

# 5. Gráficos de diagnóstico dos resíduos
ajuste.plot_diagnostics(figsize=(10, 7))
plt.tight_layout()
plt.show()

residuos = ajuste.resid.dropna()
print(acorr_ljungbox(residuos, lags=[12, 24], return_df=True))

#Previsão (12 meses) e métricas de acurácia (MAE, RMSE, MAPE)

# 1. Fazendo a previsão para os 12 meses de teste
previsao_obj = ajuste.get_forecast(steps=n_teste)
y_previsto = previsao_obj.predicted_mean
intervalo_conf = previsao_obj.conf_int(alpha=0.05)  # Intervalo de 95%

# 2. Calculando as métricas de erro fora da amostra (usando apenas NumPy)
mae = np.mean(np.abs(y_teste - y_previsto))
rmse = np.sqrt(np.mean((y_teste - y_previsto) ** 2))
mape = np.mean(np.abs((y_teste - y_previsto) / y_teste)) * 100

print("Métricas de Desempenho (Teste - 12 meses):")
print(f"MAE  (Erro Absoluto Médio):     {mae:.4f}")
print(f"RMSE (Raiz do Erro Quadrático): {rmse:.4f}")
print(f"MAPE (Erro Percentual Médio):   {mape:.2f}%")
# Previsão ingênua (repete o último valor), para comparar com o MAPE do SARIMA
prev_ing = pd.Series(y_treino.iloc[-1], index=y_teste.index)
print(f"MAPE ingênuo: {np.mean(np.abs((y_teste - prev_ing) / y_teste)) * 100:.2f}%")

# 3. Gráfico Final: Real vs. Previsto (com foco nos últimos 2 anos de treino + 1 ano de teste)
plt.figure(figsize=(12, 5))
plt.plot(y_treino.iloc[-24:], label="Treino (Últimos 2 anos)", color="gray")
plt.plot(
    y_teste.index, y_teste, label="Real (Teste 12 meses)", color="blue", marker="o"
)
plt.plot(
    y_teste.index,
    y_previsto,
    label="Previsão SARIMA(1,1,1)(0,1,1)12",
    color="red",
    linestyle="--",
    marker="x",
)
plt.fill_between(
    y_teste.index,
    intervalo_conf.iloc[:, 0],
    intervalo_conf.iloc[:, 1],
    color="pink",
    alpha=0.3,
    label="Intervalo de Previsão 95%",
)

plt.title("Previsão da Inflação EUA - SARIMA(1,1,1)(0,1,1)12")
plt.xlabel("Data")
plt.ylabel("Índice de Inflação")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()