# dFCAnalysis

Example usage for building initial models: 
```
python pipeline.py imagingData.npy targetVariable.csv subjectList.npy totalSubsToRun Resamples workDir/ windowLength dynamicOrStatic outputLabel --corrtype partial/pearson --confound confounds.csv --window_anchor middle --cpmconfig configFileBuild.yaml
```

Example usage for testing models:

```
python testModelsPara.py configFileTest.yaml
```
