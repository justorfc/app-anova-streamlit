import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 1. Configuración de la página
st.set_page_config(page_title="Análisis ANOVA", layout="centered")

st.title("📊 Análisis de Varianza (ANOVA) de un factor")
st.write("""
Sube un archivo CSV donde cada columna represente un tratamiento distinto 
(formato de libro de texto). La aplicación se encargará de reestructurar 
los datos y calcular la tabla ANOVA.
""")

# 2. Widget para cargar el archivo
archivo_subido = st.file_uploader("Sube tu archivo CSV aquí", type=["csv"])

# 3. Lógica principal: Si el usuario sube un archivo
if archivo_subido is not None:
    # Leer el CSV
    df_ancho = pd.read_csv(archivo_subido)
    
    st.subheader("1. Datos Originales (Formato Ancho)")
    st.write("Así es como se ven los datos ingresados:")
    st.dataframe(df_ancho, use_container_width=True)
    
    # Transformar los datos usando pd.melt
    df_largo = pd.melt(
        df_ancho,
        var_name='Tratamiento', 
        value_name='Respuesta'
    )
    
    # Eliminar valores nulos en caso de que los tratamientos tengan distinto número de réplicas
    df_largo = df_largo.dropna()
    
    st.subheader("2. Datos Transformados (Formato Largo)")
    st.write("Datos reestructurados en dos columnas para el motor estadístico:")
    st.dataframe(df_largo, use_container_width=True)
    
    st.divider() # Línea divisoria visual
    
    # 4. Ajuste del modelo y cálculo del ANOVA
    st.subheader("3. Resultados del ANOVA")
    
    try:
        # Ajustamos el modelo lineal
        modelo = ols('Respuesta ~ C(Tratamiento)', data=df_largo).fit()
        # Generamos la tabla ANOVA
        tabla_anova = sm.stats.anova_lm(modelo, typ=2)
        
        # Mostramos la tabla de resultados
        st.dataframe(tabla_anova, use_container_width=True)
        
        # Pequeña interpretación automática basada en el valor p
        p_valor = tabla_anova.loc['C(Tratamiento)', 'PR(>F)']
        if p_valor < 0.05:
            st.success(f"**Conclusión:** El valor *p* es {p_valor:.4f} (< 0.05). Hay diferencias significativas entre al menos dos tratamientos.")
        else:
            st.info(f"**Conclusión:** El valor *p* es {p_valor:.4f} (>= 0.05). No hay evidencia suficiente para afirmar que los tratamientos son diferentes.")
            
    except Exception as e:
        st.error(f"Hubo un error al calcular el ANOVA. Verifica que tus datos sean numéricos. Error técnico: {e}")

else:
    st.info("👆 Esperando a que subas un archivo CSV para comenzar el análisis.")