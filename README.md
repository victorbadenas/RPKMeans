# MAI-URL

[An efficient approximation to the K-means clustering for Massive Data](https://doi.org/10.1016/j.knosys.2016.06.031)  implementation for Unsupervised and Reinforcement Learning class in Master of Artificial Intelligence (MAI-UPC) at FIB

```bibtex
@article{CAPO201756,
    title = {An efficient approximation to the K-means clustering for massive data},
    journal = {Knowledge-Based Systems},
    volume = {117},
    pages = {56-69},
    year = {2017},
    note = {Volume, Variety and Velocity in Data Science},
    issn = {0950-7051},
    doi = {https://doi.org/10.1016/j.knosys.2016.06.031},
    url = {https://www.sciencedirect.com/science/article/pii/S0950705116302027},
    author = {Marco Capó and Aritz Pérez and Jose A. Lozano},
    keywords = {-means, Clustering, -means++, Minibatch -means},
    abstract = {Due to the progressive growth of the amount of data available in a wide variety of scientific fields, it has become more difficult to manipulate and analyze such information. In spite of its dependency on the initial settings and the large number of distance computations that it can require to converge, the K-means algorithm remains as one of the most popular clustering methods for massive datasets. In this work, we propose an efficient approximation to the K-means problem intended for massive data. Our approach recursively partitions the entire dataset into a small number of subsets, each of which is characterized by its representative (center of mass) and weight (cardinality), afterwards a weighted version of the K-means algorithm is applied over such local representation, which can drastically reduce the number of distances computed. In addition to some theoretical properties, experimental results indicate that our method outperforms well-known approaches, such as the K-means++ and the minibatch K-means, in terms of the relation between number of distance computations and the quality of the approximation.}
}
```

## datasets

Dataset retrieved from https://archive.ics.uci.edu/ml/machine-learning-databases/00322/data.zip:

```bash
# download and extract
python scripts/download_data.py -d https://archive.ics.uci.edu/ml/machine-learning-databases/00322/data.zip -n gas_sensor

# normalize format to csv
python scripts/format_gas_sensor_data.py
```

The additional real world datasets were retrieved from https://github.com/deric/clustering-benchmark/tree/master/src/main/resources/datasets/real-world and then converted with the `scripts/data/arffToCsv.py` script

- The `yeast` dataset was removed due to complications with using spaces instead of commas as separators
- The `wine` dataset was removed due to having the class as the first instead of last column
- The `water-treatment` was removed due to the same reason.
- The `wdbc` was removed due to the same reason.
