# Análisis de los Determinantes de la Abstención Electoral en España 🗳️

Este proyecto analiza los factores demográficos, económicos y territoriales que influyen en el comportamiento electoral en los municipios de España. [cite_start]Fue desarrollado como parte del **Máster en Big Data, Data Science & Inteligencia Artificial** de la Universidad Complutense de Madrid[cite: 2001].

## 📋 Descripción del Proyecto

La participación electoral es un indicador clave de la salud democrática. [cite_start]Este estudio utiliza técnicas de **Minería de Datos** y **Modelización Predictiva** para abordar dos objetivos analíticos complementarios basados en datos reales de municipios españoles[cite: 2003, 2011]:

1.  **Regresión Lineal (Enfoque Explicativo):** Modelizar el porcentaje exacto de abstención (`AbstentionPtge`) en función de variables socioeconómicas.
2.  [cite_start]**Regresión Logística (Enfoque de Clasificación):** Predecir la probabilidad de que un municipio tenga una "Abstención Alta" (superior al 30%) (`AbstencionAlta`)[cite: 2008].

El flujo de trabajo abarca desde la depuración de datos crudos hasta la selección de variables mediante algoritmos *Stepwise* y la validación cruzada de los modelos.

## 🛠️ Metodología

El proyecto sigue un pipeline riguroso de Data Science:

### 1. Ingeniería y Limpieza de Datos
[cite_start]Se realizó un preprocesamiento exhaustivo para garantizar la calidad del dato[cite: 2014, 2024]:
* **Tratamiento de Nulos:** Imputación estadística (mediana/moda) y análisis de patrones de correlación de valores ausentes.
* **Detección de Anomalías:** Análisis de *outliers* mediante rango intercuartílico y visualización robusta.
* **Corrección de Errores:** Estandarización de variables categóricas (CCAA, Actividad Principal) y corrección de inconsistencias numéricas.

![Análisis de Valores Perdidos](grafico_missings_filtrado.png)
*Figura 1: Análisis de la calidad del dato previo al modelado.*

### 2. Modelado Predictivo
[cite_start]Se implementaron algoritmos de selección de variables clásica (Forward, Backward, Stepwise) optimizando criterios de información (AIC/BIC)[cite: 2027, 2037].

* **Modelo Lineal:** Capaz de explicar un 36% de la varianza del fenómeno ($R^2 \approx 0.36$), destacando la importancia de factores regionales.
* **Modelo Logístico:** Alcanzó un **AUC > 0.80** en el conjunto de test, demostrando una gran capacidad discriminante para detectar municipios con riesgo de alta abstención.

![Distribución de Atípicos](grafico_atipicos.png)
*Figura 2: Distribución de variables y detección de valores atípicos normalizados.*

## 📊 Tecnologías Utilizadas

* **Python 3.x**
* **Pandas & NumPy:** Manipulación algebraica y de datos.
* **Scikit-learn:** Modelado (Regresión Lineal, Logística), selección de variables y métricas de evaluación.
* **Statsmodels:** Inferencia estadística detallada.
* **Matplotlib & Seaborn:** Visualización de datos.

## 🚀 Cómo ejecutar el código

1.  Clona este repositorio:
    ```bash
    git clone [https://github.com/TU_USUARIO/analisis-elecciones-espana.git](https://github.com/TU_USUARIO/analisis-elecciones-espana.git)
    ```
2.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ejecuta el script principal:
    ```bash
    python src/main_analisis_elecciones.py
    ```

## 📄 Autor
**Juan Peñas Utrilla**
Máster en Big Data, Data Science & Inteligencia Artificial.

---
*Nota: Los datos utilizados son propiedad de sus respectivas fuentes y se utilizan aquí con fines académicos.*
