import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io.arff import loadarff
from .utils import bytesToString
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler, Normalizer, LabelEncoder, OneHotEncoder

# NumPy Data types
CATEGORICAL = 'O'

# String Conversion Methods
INT = 'int'
ONEHOT = 'onehot'
STRING_CONVERSION = [INT, ONEHOT]

# Float Normalization Methods
MIN_MAX = "min-max"
STANDARISATION = "stardardisation"
MEAN = "mean"
UNIT = "unit"
FLOAT_NORMALIZATION = [MIN_MAX, STANDARISATION, MEAN, UNIT]

# Missing Data Imputation Methods
IMPUTE_MOST_FREQUENT = "most_frequent"
IMPUTE_MEDIAN = "median"
IMPUTE_MEAN = "mean"
IMPUTE_CONSTANT = "constant"
RANDOM = "random"
LINEAR_REGRESSION = "linear_regression"
LOGISTIC_REGRESSION = "logistic_regression"
MISSING_DATA_IMPUTATION = [IMPUTE_MOST_FREQUENT, IMPUTE_MEDIAN, IMPUTE_MEAN,
                           IMPUTE_CONSTANT, RANDOM, LINEAR_REGRESSION, LOGISTIC_REGRESSION]

class ArffFile:
    """
    Class responsible of loading and formatting the arff file data

    input:
        args:
            arffPath: Path object with information on the path to arffFile

    attributes:
        path: path to arff file
        data: pd.DataFrame with arff data in float values (normalised in range [0, 1])
        metaData: arff metadata information
        maps: dictionary with mapping information from string to int or OneHotVector

    """
    def __init__(self, arffPath, stringConversion=INT, floatNormalization=MIN_MAX, missingDataImputation=IMPUTE_MOST_FREQUENT):
        self.assertInitParameters(stringConversion, floatNormalization, missingDataImputation)
        self.path = arffPath
        self.stringConversion = stringConversion
        self.floatNormalization = floatNormalization
        self.missingDataImputation = missingDataImputation
        self.labelEncoders = {}
        data, self.metaData = loadarff(arffPath)
        self.formatDataFrame(data)

    def assertInitParameters(self, stringConversion, floatNormalization, missingDataImputation):
        assert stringConversion in STRING_CONVERSION, f"{stringConversion} is not a valid value for stringConversion"
        assert floatNormalization in FLOAT_NORMALIZATION, f"{floatNormalization} normalization type not supported"
        assert missingDataImputation in MISSING_DATA_IMPUTATION, f"{missingDataImputation} is not supported"

    def formatDataFrame(self, arffData):
        data = pd.DataFrame(arffData)
        self.data = data.applymap(bytesToString) # apply type conversion to all items in DataFrame
        self.formatColumns()

    def formatColumns(self):
        columnTypes = {}
        columnNames = self.data.columns.to_list()
        for column in self.data.columns[:-1].to_list():
            columnData = self.data[column].copy()
            columnTypes[column] = columnData.dtype.kind
            self.preprocessDataTypes(column, columnData, columnNames)
        for column in self.data.columns[:-1].to_list():
            self.preprocessDataMissingValues(column, columnTypes[column], columnNames)
            self.preprocessDataRanges(column, columnTypes[column])
        self.formatGoldStandard(columnNames[-1])

    def preprocessDataTypes(self, column, columnData, columnNames):
        if columnData.dtype.kind == CATEGORICAL:
            self.formatCategoricalData(column, columnData, columnNames)

    def preprocessDataRanges(self, column, columnType):
        if not (columnType == CATEGORICAL and self.stringConversion == ONEHOT):
            self.data[column] = self.normalizeFloatColumn(self.data[column])

    def preprocessDataMissingValues(self, column, columnType, columnNames):
        if columnType == CATEGORICAL:
            if self.stringConversion == INT:
                if '?' in self.labelEncoders[column].classes_:
                    missingValue = self.labelEncoders[column].transform(["?"])[0]
                    self.data[column] = self.fixMissingValues(column, self.data[column], columnNames, missingValue)

    def formatGoldStandard(self, gsColumn):
        self.data[gsColumn] = self.convertStringsToInt(gsColumn, self.data[gsColumn])

    def formatCategoricalData(self, column, columnData, columnNames):
        if self.stringConversion == INT:
            self.data[column] = self.convertStringsToInt(column, columnData)
        elif self.stringConversion == ONEHOT:
            self.data = self.convertStringToOH(column, columnData, columnNames)

    def fixMissingValues(self, column, columnData, columnNames, missingValue):
        if self.missingDataImputation == RANDOM:
            return self.randomMissingValues(columnData, missingValue)
        elif self.missingDataImputation in [IMPUTE_MOST_FREQUENT, IMPUTE_MEDIAN, IMPUTE_MEAN, IMPUTE_CONSTANT]:
            return self.imputerMissingValues(columnData, missingValue)
        elif self.missingDataImputation in [LINEAR_REGRESSION, LOGISTIC_REGRESSION]:
            return self.regressionMissingValues(column, columnData, columnNames, missingValue)

    def randomMissingValues(self, columnData, missingValue):
        columnData[columnData == missingValue] = np.random.choice(columnData[columnData != missingValue],
                                                         columnData[columnData == missingValue].count())
        return columnData

    def imputerMissingValues(self, columnData, missingValue):
        columnData = SimpleImputer(missing_values=missingValue, strategy=self.missingDataImputation) \
                    .fit_transform(columnData.values.reshape(-1, 1))
        return columnData.squeeze()

    def regressionMissingValues(self, column, columnData, columnNames, missingValue):
        model = LogisticRegression(n_jobs=-1) if self.missingDataImputation == LOGISTIC_REGRESSION \
                                              else LinearRegression(n_jobs=-1)
        features = list(set(columnNames[:-1]) - set(column))
        allOtherColumnsData = self.data[features]
        model.fit(X=allOtherColumnsData[columnData != missingValue], y=columnData[columnData != missingValue])
        columnData[columnData == missingValue] = model.predict(allOtherColumnsData[columnData == missingValue])
        return columnData

    def convertStringsToInt(self, column, columnData):
        self.labelEncoders[column] = LabelEncoder()
        return self.labelEncoders[column].fit_transform(columnData).astype(np.int)

    def convertStringToOH(self, column, columnData, columnNames):
        self.labelEncoders[column] = OneHotEncoder(sparse=False, handle_unknown='ignore')
        oneHotVectors = self.labelEncoders[column].fit_transform(columnData.reshape(-1, 1)).astype(np.int)
        OHColumnNames = [f"{column}_{columnValue}" for columnValue in self.labelEncoders[column].categories_[0]]
        OHDataFrame = pd.DataFrame(oneHotVectors, columns=OHColumnNames)
        columnNames[columnNames.index(column):columnNames.index(column)+1] = OHDataFrame.columns.to_list()
        return self.data.drop(column, axis=1).join(OHDataFrame, how='outer')[columnNames]

    def normalizeFloatColumn(self, data):
        if self.floatNormalization == STANDARISATION:
            return scale(data)
        elif self.floatNormalization == MEAN:
            scaler = StandardScaler()
        elif self.floatNormalization == MIN_MAX:
            scaler = MinMaxScaler()
        elif self.floatNormalization == UNIT:
            scaler = Normalizer()
        data = np.array(data).reshape(-1, 1)
        data = scaler.fit_transform(data)
        return pd.Series(data.reshape(-1))

    def getData(self):
        return self.data

    def getMetaData(self):
        return self.metaData

    def getStringMapData(self):
        return self.labelEncoders

if __name__ == "__main__":
    arffFile = ArffFile(Path("../datasets/adult.arff"))
    print(arffFile.getData().head())
    arffFile.scatterPlot(figsize=(10, 10))
