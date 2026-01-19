# -*- coding: utf-8 -*-
"""
Proyecto: Análisis de Datos Elecciones España

"""


# 1. LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

# Importación de funciones propias (FuncionesMineria.py debe estar en el mismo directorio)
from FuncionesMineria import (
    analizar_variables_categoricas, cuentaDistintos, frec_variables_num,
    atipicosAmissing, patron_perdidos, ImputacionCuant, ImputacionCuali,
    graficoVcramer, mosaico_targetbinaria, boxplot_targetbinaria,
    hist_targetbinaria, Transf_Auto, lm, Rsq, validacion_cruzada_lm,
    modelEffectSizes, crear_data_modelo, Vcramer,
    lm_stepwise, lm_backward, lm_forward,
    glm_stepwise, glm_backward, glm_forward, impVariablesLog,
    summary_glm, validacion_cruzada_glm, sensEspCorte, curva_roc,pseudoR2
)

plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")

# 2. IMPORTACIÓN DE DATOS

datos_original = pd.read_excel("DatosEleccionesEspaña.xlsx")

# Eliminación de columnas no útiles o no permitidas para los modelos
columnas_eliminar = ['Izda_Pct', 'Dcha_Pct', 'Otros_Pct', 'Izquierda', 
                     'Derecha', 'Name', 'CodigoProvincia']
datos = datos_original.drop(columns=columnas_eliminar)

# Corrección inicial de tipos (AbstencionAlta se trata inicialmente como categórica para descriptivos)
datos['AbstencionAlta'] = datos['AbstencionAlta'].astype(str)

# Separación de tipos de variables
variables = list(datos.columns)
numericas = datos.select_dtypes(include=['int', 'int32', 'int64', 'float', 'float32', 'float64']).columns
categoricas = [variable for variable in variables if variable not in numericas]

# 3. ANÁLISIS DESCRIPTIVO Y CORRECCIÓN DE ERRORES

# 3.1 Corrección CCAA (Agrupación de categorías poco representadas < 100)
freq_ccaa = datos['CCAA'].value_counts()
ccaa_poco_repr = freq_ccaa[freq_ccaa < 100].index
datos['CCAA'] = datos['CCAA'].replace(ccaa_poco_repr, 'Can_Ast_Bal_Mur')

# 3.2 Corrección Actividad Principal (Agrupación)
nueva_actividad = {
    'Servicios': 'Servicios_Constr_Industria',
    'Construccion': 'Servicios_Constr_Industria',
    'Industria': 'Servicios_Constr_Industria'
}
datos['ActividadPpal'] = datos['ActividadPpal'].replace(nueva_actividad)

# 3.3 Corrección Explotaciones (Código 99999 a NaN)
datos['Explotaciones'] = datos['Explotaciones'].replace(99999, np.nan)

# 3.4 Corrección Porcentajes (Valores fuera de rango 0-100 a NaN)
cols_ptge = datos.filter(regex=r'Ptge$').columns.tolist()
cols_ptge.extend(["Age_19_65_pct", "Age_over65_pct"])

for columna in cols_ptge:
    datos[columna] = [x if 0 <= x <= 100 else np.nan for x in datos[columna]]

# 3.5 Corrección Densidad ('?' a NaN)
datos['Densidad'] = datos['Densidad'].replace('?', np.nan)

# Análisis descriptivo numérico tras correcciones
descriptivos_num = datos.describe().T
for num in numericas:
    descriptivos_num.loc[num, "Asimetria"] = datos[num].skew()
    descriptivos_num.loc[num, "Kurtosis"] = datos[num].kurtosis()
    descriptivos_num.loc[num, "Rango"] = np.ptp(datos[num].dropna().values)


# 4. PREPARACIÓN DE DATOS INPUT 

# Definimos targets
varObjCont = datos['AbstentionPtge']
varObjBin = datos['AbstencionAlta'] 

# Datos input (sin targets)
datos_input = datos.drop(['AbstentionPtge', 'AbstencionAlta'], axis=1)

variables_input = list(datos_input.columns)
numericas_input = datos_input.select_dtypes(include=[np.number]).columns
categoricas_input = [variable for variable in variables_input if variable not in numericas_input]

# 5. ANÁLISIS DE VALORES ATÍPICOS

resultados_atipicos = []
for col in numericas_input:
    # atipicosAmissing devuelve [serie_limpia, numero_atipicos, valores_atipicos]
    _, n_atipicos, _ = atipicosAmissing(datos_input[col])
    resultados_atipicos.append({
        'Variable': col,
        'Num_Atipicos': n_atipicos,
        'Porcentaje': round((n_atipicos / len(datos_input)) * 100, 2)
    })

df_atipicos = pd.DataFrame(resultados_atipicos).sort_values('Num_Atipicos', ascending=False)
print(df_atipicos.head())

# Gráfico de cajas normalizado (Escala 0-1) para visualizar atípicos
datos_norm = datos_input[numericas_input].copy()
datos_norm = (datos_norm - datos_norm.min()) / (datos_norm.max() - datos_norm.min())
datos_melted = datos_norm.melt(var_name='Variable', value_name='Valor Normalizado')

plt.figure(figsize=(10, 12))
sns.boxplot(data=datos_melted, x='Valor Normalizado', y='Variable', orient='h', palette='viridis', linewidth=1)
plt.title('Distribución de Variables y Valores Atípicos (Escala Normalizada 0-1)')
plt.xlabel('Rango Relativo')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('grafico_atipicos.svg', bbox_inches='tight')
plt.show()

# 6. ANÁLISIS E IMPUTACIÓN DE VALORES PERDIDOS

# Gráfico de correlación de perdidos
patron_perdidos(datos_input, threshold=0.01)

# Gráfico de porcentaje de nulos
prop_missingsVars = datos_input.isna().sum() / len(datos_input)
vars_con_missings = prop_missingsVars[prop_missingsVars > 0].sort_values(ascending=False)
vars_con_missings_pct = vars_con_missings * 100

plt.figure(figsize=(12, 6))
bars = plt.bar(vars_con_missings_pct.index, vars_con_missings_pct.values, color='blue', edgecolor='black', alpha=0.8)
plt.title('Porcentaje de Valores Perdidos por Variable', fontsize=14)
plt.ylabel('% de Nulos')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('grafico_missings_filtrado.svg')
plt.show()

# Eliminación de variables con > 50% nulos (si las hubiera)
eliminar = [prop_missingsVars.index[x] for x in range(len(prop_missingsVars)) if prop_missingsVars[x] > 0.5]
if eliminar:
    print(f"Eliminando columnas con >50% nulos: {eliminar}")
    datos_input = datos_input.drop(eliminar, axis=1)

# Imputación
# Numéricas -> Mediana (excepto PersonasInmueble que se recalcula)
for x in numericas_input:
    if x == "PersonasInmueble":
        continue
    datos_input[x] = ImputacionCuant(datos_input[x], 'mediana')

# Categóricas -> Moda
for x in categoricas_input:
    datos_input[x] = ImputacionCuali(datos_input[x], 'moda')

# Recálculo de PersonasInmueble
mask_recalc = datos_input["inmuebles"].notna() & (datos_input["inmuebles"] > 0)
datos_input.loc[mask_recalc, "PersonasInmueble"] = datos_input.loc[mask_recalc, "Population"] / datos_input.loc[mask_recalc, "inmuebles"]

# Chequeo final
print("Nulos restantes:", datos_input.isna().sum().sum())


# 7. REGRESIÓN LINEAL (Target: AbstentionPtge)

# Split Train/Test
x_train, x_test, y_train, y_test = train_test_split(datos_input, np.ravel(varObjCont), 
                                                    test_size=0.2, random_state=123456)

var_cont_lm = x_train.select_dtypes(include=[np.number]).columns.tolist()
var_categ_lm = x_train.select_dtypes(exclude=[np.number]).columns.tolist()

# Entrenamiento de modelos (Selección clásica)
mStepAIC = lm_stepwise(y_train, x_train, var_cont_lm, var_categ_lm, [], 'AIC')
mStepBIC = lm_stepwise(y_train, x_train, var_cont_lm, var_categ_lm, [], 'BIC')
mBackAIC = lm_backward(y_train, x_train, var_cont_lm, var_categ_lm, [], 'AIC')
mBackBIC = lm_backward(y_train, x_train, var_cont_lm, var_categ_lm, [], 'BIC')
mForwAIC = lm_forward(y_train, x_train, var_cont_lm, var_categ_lm, [], 'AIC')
mForwBIC = lm_forward(y_train, x_train, var_cont_lm, var_categ_lm, [], 'BIC')

modelos_lm = {
    "Stepwise AIC": mStepAIC, "Stepwise BIC": mStepBIC,
    "Backward AIC": mBackAIC, "Backward BIC": mBackBIC,
    "Forward AIC":  mForwAIC, "Forward BIC":  mForwBIC,
}

# Tabla Resumen Lineal
resumen_filas = []
for nombre, m in modelos_lm.items():
    r2_train = Rsq(m['Modelo'], y_train, m['X'])
    x_test_m = crear_data_modelo(x_test, m['Variables']['cont'], m['Variables']['categ'], [])
    r2_test  = Rsq(m['Modelo'], y_test, x_test_m)
    n_param = int(m['Modelo'].df_model + 1)
    
    resumen_filas.append({"Modelo": nombre, "R2_Train": r2_train, "R2_Test": r2_test, "N_Param": n_param})

tabla_lm = pd.DataFrame(resumen_filas).sort_values("R2_Test", ascending=False)
print("Resultados Regresión Lineal:")
print(tabla_lm.round(4))
tabla_lm.to_excel("resultados_lineal.xlsx", index=False)

# Validación Cruzada Lineal y Boxplot
results_cv_lm = pd.DataFrame({'Rsquared': [], 'Resample': [], 'Modelo': []})
for rep in range(20): # Según PDF son 20 reps
    for nombre, m in modelos_lm.items():
        cv_scores = validacion_cruzada_lm(5, x_train, y_train, 
                                          m['Variables']['cont'], m['Variables']['categ'], [])
        temp_df = pd.DataFrame({
            'Rsquared': cv_scores,
            'Resample': ['Rep' + str(rep + 1)] * 5,
            'Modelo': [nombre] * 5
        })
        results_cv_lm = pd.concat([results_cv_lm, temp_df], axis=0)

plt.figure(figsize=(10, 6))
sns.boxplot(data=results_cv_lm, x='Modelo', y='Rsquared', showfliers=False)
sns.stripplot(data=results_cv_lm, x='Modelo', y='Rsquared', jitter=False, color='black', alpha=0.5)
plt.title("Distribución del R² por modelo (CV Repetida - Lineal)")
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('boxplot_regresion_lineal.svg')
plt.show()

# Modelo Ganador Lineal: Stepwise BIC (según PDF)
modelo_ganador_lm = mStepBIC
print(modelo_ganador_lm['Modelo'].summary())

# Importancia de variables
importancia_lineal = modelEffectSizes(modelo_ganador_lm['Modelo'], y_train, x_train,                               
                                      modelo_ganador_lm['Variables']['cont'],
                                      modelo_ganador_lm['Variables']['categ'])


# 8. REGRESIÓN LOGÍSTICA (Target: AbstencionAlta)

# Split Train/Test (Convertir target a int para sklearn)
x_train_log, x_test_log, y_train_log, y_test_log = train_test_split(
    datos_input, np.ravel(varObjBin), test_size=0.2, random_state=13
)
y_train_log = y_train_log.astype(int)
y_test_log = y_test_log.astype(int)

var_cont_log = x_train_log.select_dtypes(include=[np.number]).columns.tolist()
var_categ_log = x_train_log.select_dtypes(exclude=[np.number]).columns.tolist()

# Entrenamiento de modelos
mStepAIC_log = glm_stepwise(y_train_log, x_train_log, var_cont_log, var_categ_log, [], 'AIC')
mStepBIC_log = glm_stepwise(y_train_log, x_train_log, var_cont_log, var_categ_log, [], 'BIC')
mBackAIC_log = glm_backward(y_train_log, x_train_log, var_cont_log, var_categ_log, [], 'AIC')
mBackBIC_log = glm_backward(y_train_log, x_train_log, var_cont_log, var_categ_log, [], 'BIC')
mForwAIC_log = glm_forward(y_train_log, x_train_log, var_cont_log, var_categ_log, [], 'AIC')
mForwBIC_log = glm_forward(y_train_log, x_train_log, var_cont_log, var_categ_log, [], 'BIC')

modelos_log = {
    "Stepwise AIC": mStepAIC_log, "Stepwise BIC": mStepBIC_log,
    "Backward AIC": mBackAIC_log, "Backward BIC": mBackBIC_log,
    "Forward AIC":  mForwAIC_log, "Forward BIC":  mForwBIC_log,
}

# Tabla Comparativa Logística y Validación Cruzada
K = 5
REPS = 20
results_auc = [] # Para boxplot
filas_log = []   # Para tabla

for nombre, m in modelos_log.items():
    # AUC Test
    x_test_m = crear_data_modelo(x_test_log, m['Variables']['cont'], m['Variables']['categ'], [])
    auc_test = roc_auc_score(y_test_log, m['Modelo'].predict_proba(x_test_m)[:, 1])
    
    # AUC CV Repetida
    aucs_all = []
    for rep in range(REPS):
        aucs = validacion_cruzada_glm(K, x_train_log, y_train_log,
                                      m['Variables']['cont'], m['Variables']['categ'], [])
        for auc in aucs:
            results_auc.append({"Modelo": nombre, "AUC_CV": auc, "Rep": rep + 1})
        aucs_all.extend(aucs)
    
    auc_cv_mean = float(np.mean(aucs_all))
    pr2_train = pseudoR2(m['Modelo'], m['X'], y_train_log) 
    n_param = len(m['Modelo'].coef_[0]) + 1
    
    filas_log.append([nombre, auc_test, auc_cv_mean, pr2_train, n_param])

tabla_log = pd.DataFrame(filas_log, columns=["Modelo", "AUC_Test", "AUC_CV", "PseudoR2_Train", "N_param"])
tabla_log = tabla_log.sort_values(["AUC_CV", "AUC_Test"], ascending=False)
print("Resultados Regresión Logística:")
print(tabla_log.round(4))
tabla_log.round(4).to_excel("resultados_logistica.xlsx", index=False)

# Boxplot Logística
df_auc = pd.DataFrame(results_auc)
plt.figure(figsize=(10,5))
sns.boxplot(data=df_auc, x="Modelo", y="AUC_CV", showfliers=True)
sns.stripplot(data=df_auc, x="Modelo", y="AUC_CV", color="black", alpha=0.6, jitter=False)
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.title("Distribución de AUC por modelo (CV - Logística)")
plt.savefig("grafico_modelos_logistica.svg", bbox_inches="tight")
plt.show()

# Modelo Ganador Logística: 
modelo_ganador_log = mStepBIC_log

var_cont_g = modelo_ganador_log['Variables']['cont']
var_categ_g = modelo_ganador_log['Variables']['categ']
var_inter_g = modelo_ganador_log['Variables'].get('inter', [])

# Curva ROC Ganador
x_test_m_g = crear_data_modelo(x_test_log, var_cont_g, var_categ_g, var_inter_g)
auc_test_g = curva_roc(x_test_m_g, y_test_log, modelo_ganador_log)

# Punto de corte óptimo (Youden)
grid = np.linspace(0.01, 0.99, 99)
mejor_p, mejor_J, mejor_met = None, -999, None

for p in grid:
    met = sensEspCorte(modelo_ganador_log['Modelo'], x_test_log, y_test_log, float(p),
                       var_cont_g, var_categ_g, var_inter_g)
    sens = float(met["Sensitivity"].iloc[0])
    spec = float(met["Specificity"].iloc[0])
    J = sens + spec - 1
    if J > mejor_J:
        mejor_J, mejor_p, mejor_met = J, float(p), met

print(f"Punto de corte óptimo (Youden): {round(mejor_p, 4)}")
print(mejor_met.round(4))

# Matriz de confusión con corte óptimo
prob = modelo_ganador_log['Modelo'].predict_proba(x_test_m_g)[:,1]
yhat = (prob > mejor_p).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test_log, yhat).ravel()
conf_matrix_df = pd.DataFrame([[tn, fp],[fn, tp]], index=["Real 0","Real 1"], columns=["Pred 0","Pred 1"])
print("\nMatriz de Confusión:")
print(conf_matrix_df)

# Summary y Odds Ratios
X_modelo_float = modelo_ganador_log['X'].astype(float)
resumen_log = summary_glm(modelo_ganador_log['Modelo'], y_train_log, X_modelo_float)
coefs = resumen_log['Contrastes'].copy()
coefs['Odds Ratio'] = np.exp(coefs['Estimate'])

print("\nCoeficientes y Odds Ratios:")
print(coefs)

imp = impVariablesLog(modelo_ganador_log, y_train_log, x_train_log, 
                          var_cont_g, var_categ_g, var_inter_g)

