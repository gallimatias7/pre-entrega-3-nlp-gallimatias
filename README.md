# Pre-entrega 3: Clasificador Supervisado de Texto con TF-IDF

Pipeline de clasificación de texto supervisado sobre el corpus **AG News** (8.000 muestras de entrenamiento y 2.000 de prueba), desarrollado para el Módulo 3 de Data Science III (Coderhouse).

---

## 🛠️ Entorno y Reproducibilidad
* **Python:** 3.10+
* **Librerías principales:** `scikit-learn==1.6.1`, `pandas==2.2.2`, `numpy==2.1.3`, `spacy==3.8.5`, `matplotlib==3.10.0`, `seaborn==0.13.2`

Instalación rápida:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Decisiones de Ingeniería y Experimentación

### 1. Vectorización con TF-IDF y Control de Data Leakage
Para evitar fuga de información (*Data Leakage*), el vectorizador se ajustó mediante `fit_transform` exclusivamente sobre el conjunto de entrenamiento (`train_df`). El conjunto de evaluación (`test_df`) fue transformado de forma ciega mediante `.transform()`.

Se experimentó con la dimensionalidad léxica evaluando combinaciones de N-gramas y límite de vocabulario:
* **Unigramas (3.000 features):** F1-Macro = 0.8910
* **Unigramas (5.000 features):** F1-Macro = 0.8950
* **Unigramas + Bigramas (5.000 features):** F1-Macro = 0.8951
* **Unigramas + Bigramas (10.000 features):** F1-Macro = 0.8945
* **Unigramas + Bigramas (Sin restricción, 162.337 features):** F1-Macro = 0.8964

**Configuración seleccionada:** `ngram_range=(1, 2)` con `max_features=5000`. Limitar el espacio a 5.000 características retiene las combinaciones clave (`oil prices`, `prime minister`, `red sox`) y alcanza un F1 idéntico al modelo no restringido (0.8951 vs. 0.8964), reduciendo en más de un 96% el tamaño de la matriz en memoria RAM y previniendo sobreajuste por ruido de baja frecuencia.

### 2. Comparación y Elección del Clasificador
Entrenando sobre la matriz balanceada de 5.000 características, se evaluaron tres modelos supervisados:
1. **Multinomial Naive Bayes (alpha=0.1):** F1-Macro = 0.8902 | Accuracy = 89.00%
2. **Linear Support Vector Classifier (LinearSVC):** F1-Macro = 0.8871 | Accuracy = 88.70%
3. **Regresión Logística (L2, max_iter=1000):** **F1-Macro = 0.8951 | Accuracy = 89.50%**

**Justificación:** Se seleccionó **Regresión Logística**. En espacios de alta dimensionalidad derivados de TF-IDF, el hiperplano lineal ofrece una separación óptima con convergencia estable, regularización L2 integrada y la capacidad de emitir probabilidades calibradas (`predict_proba`), superando en rendimiento general a Naive Bayes y SVM.

---

## 📊 Análisis de la Matriz de Confusión y Diagnóstico de Errores

### Rendimiento por Categoría (Reporte de Test)
* **Sports (F1 = 0.9551, Recall = 0.9580):** Es la clase con mayor desempeño. El vocabulario deportivo (`game`, `season`, `win`, nombres de franquicias) es altamente específico y tiene escaso solapamiento léxico con otras categorías.
* **World (F1 = 0.9028, Recall = 0.8920):** Desempeño sólido basado en entidades geopolíticas claras (`prime minister`, `un`, `treaty`, nombres de países).
* **Sci_Tech (F1 = 0.8723) y Business (F1 = 0.8500):** Son las categorías con mayor desafío de clasificación.

### Detección de Confusión Sistemática
La matriz de confusión revela que el mayor porcentaje de error del modelo se concentra de forma bidireccional entre **Business** y **Sci_Tech**:
* **46 noticias reales de Business** fueron clasificadas como Sci_Tech.
* **44 noticias reales de Sci_Tech** fueron clasificadas como Business.

**Causa analítica:** En el periodismo moderno existe una intersección semántica natural entre las noticias corporativas y el sector tecnológico. Artículos sobre compañías como Microsoft, Apple, Intel o Google integran simultáneamente métricas financieras (*stocks*, *quarterly profits*, *shares*, *acquisition*) y terminología tecnológica (*software*, *chips*, *operating system*). Un modelo lineal basado en frecuencias léxicas aisladas encuentra dificultades para discernir el foco principal del texto sin comprensión del orden sintáctico contextual.

---

## 🚀 Conclusión hacia el Módulo 4
El clasificador de Regresión Logística con TF-IDF establece una base sólida con un **F1-Macro de 0.8951**, cumpliendo el objetivo de referencia de la cátedra (~0.90). Este modelo servirá como punto de comparación (*benchmark*) en el Módulo 4, donde implementaremos un Transformer contextual (*DistilBERT*) con LoRA para evaluar si los mecanismos de autoatención logran desambiguar la frontera entre *Business* y *Sci_Tech*.
