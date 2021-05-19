# Preguntas 19-05

## Paper

- Contradiccion en el paper, top-down o dottom-up
  - up-down.
- RPKM por grid partitions o caso general? creo que en el caso del paper solo mencionan grid partitios, but not sure.
  - ok solo grid y es correcto.
- Si las centroides no son suficientes, que hacemos?
  - skip to next
- 5% overlap dataset
  - fix centers and compute variange given that set of centers to control overlap.
- Distance computations en sklearn?
  - calculo a priori
  - usan elkan pero usamos un kmeans vanilla para calcular las distancias.
  - only 1 init
  - fix seed in every run.

## Documentation

- Que secciones debe tener el report?
  - Resumen paper mejoras que propone
  - Algoritmo implementado
  - Experimentos
