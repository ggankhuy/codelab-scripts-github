import h2o
h2o.init()

df = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/prostate/prostate.csv.zip")
colmeans = df.mean()
print(colmeans)
