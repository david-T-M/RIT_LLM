# RIT_LLM
Experimentos con LLM y features RIT

##
Vamos a realizar una comparativa en dos lineas
1. Rendimiento
2. Justificación

Para lograr esto requerimos evaluar nuestras caracteristicas en los diferentes corpus para saber 
quien ayuda resolver en cada categoría de benchmarck GLUE

1. Lexical Semantics
2.Predicate-Argument Structure
3. Logic
4. Knowledge & Common Sense

      SICK  | SciTail | SNLI  | MultNLI |   Diagnostico


Clase 
ShapValue en cada categoría

De ahí podremos identificar que features ayudan en cada caso
Experimento 0. 
Utilizar los train de cada uno de los corpus para obtener valores de shap para cada categoría del corpus de diagnostico

Experimento 0.1
Utilizar unicamente el MultNLI train para posteriormente evaluar el corpus diagnostico
y obtener valores shap para cada caracteristicas



# Creación de modelos
Modelo base
./ollama create dRIT -f ./FilesModel/ModelfileRIT

Modelo proporcionando las features obtenidas
./ollama create dRIT_pi -f ./FilesModel/ModelfileRIT_pi

Modelo proporcionando las features obtenidas - few shot
./ollama create dRIT_pi -f ./FilesModel/ModelfileRIT_pi

Se usaran un muestreo de clases por 200 por clase
osea son 600

