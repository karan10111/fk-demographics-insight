package com.flipkart.demographic.gender;

/**
 * Created by karan.verma on 26/06/17.
 */


import com.flipkart.fdp.ml.export.ModelExporter;
import com.flipkart.fdp.ml.importer.ModelImporter;
import com.flipkart.fdp.ml.transformer.Transformer;
import com.flipkart.fdp.ml.utils.SchemaExporter;
import com.flipkart.mlplatform.*;
import com.flipkart.mlplatform.entities.*;
import com.flipkart.mlplatform.exceptions.MLAPIClientErrorException;
import com.flipkart.mlplatform.exceptions.MLAPIException;
import com.flipkart.mlplatform.exceptions.MLAPISDKException;
import com.flipkart.mlplatform.spark.SparkMLAPI;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.DataFrame;
import java.util.*;
import com.flipkart.fdp.ml.Log1PScaler;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import com.flipkart.mlplatform.MLAPI;
import org.apache.spark.sql.Row;

import java.util.LinkedList;
import java.util.List;

import static org.apache.spark.sql.types.DataTypes.DoubleType;


public class GenderTraining {


    private static final Gson gson = new Gson();

    public static JavaSparkContext getJavaSparkContext() {
        JavaSparkContext sparkContext = new JavaSparkContext(new SparkConf());
        return sparkContext;
    }

    public static void main(String[] args) throws Exception {
        JavaSparkContext jsc = getJavaSparkContext();
        System.setProperty("ml.platform.config.bucket", "prod-ml-platform-sdk");
        System.setProperty("HADOOP_USER_NAME", "hdfs");
        SparkMLAPI mlapi = SparkMLAPI.getInstance();

//        String hdfsPath = args[0];
//        String fileName = args[1];
//        System.out.println(hdfsPath);
//        System.out.println(fileName);
//
//        DatasetId datasetId = new DatasetId("DS3", "1.37");
        DatasetId datasetIdTest = new DatasetId("DS4", "1.21");
//
//        Model model = train(datasetId, mlapi, jsc);
//        predLabelFile(model, datasetIdTest, mlapi, jsc, hdfsPath, fileName);
        ModelId modelId = new ModelId("MD23177", "1.0.0");
        com.flipkart.mlplatform.Model model = loadModel(modelId);
        List<Input> inputs = model.getModelMetadata().getInputs();
        ArrayList<String> input_names = new ArrayList<String>();


        for(Input input : inputs){
            System.out.println(input.getName());
            input_names.add(input.getName());
        }
        System.out.println("**************model inputs above****************");
        for(String fn: input_names){
            System.out.println(fn);
        }
        System.out.println("**************got them into list****************");
        DataFrame testDataFrame  = getTrainingDataFrame(jsc, mlapi, datasetIdTest);
        Row df_row = testDataFrame.take(1)[0];
        List<String> columns = Arrays.asList(testDataFrame.columns());
        for(String col : columns){
            System.out.println(col + " " + columns.indexOf(col));
        }
        System.out.println("**************columns of df printed****************");

        HashMap<String, Object> hsmp = new HashMap<>();

        for(Input input: inputs){
            int of_feature = columns.indexOf(input.getName());
            System.out.println("of_feature");
            System.out.println(of_feature + " " + input.getName() + " " + df_row.get(of_feature));
            hsmp.put(input.getName(), Double.parseDouble(df_row.get(of_feature).toString()));
        }
        testDataFrame.show(2);
        for(String key: hsmp.keySet()){
            System.out.println(key + " " + hsmp.get(key));
        }

        PredictionInput pi = new PredictionInput();
        pi.setParams(hsmp);

        PredictionOutput po = model.predict(pi);
        System.out.println(po.getResult().get("probability"));
    }






    public static com.flipkart.mlplatform.Model loadModel(ModelId modelId) throws Exception{
        MLAPI modelApi = MLAPI.getInstance();
        return modelApi.loadModel(modelId);
    }



    public static Model train(DatasetId datasetId, SparkMLAPI mlapi, JavaSparkContext jsc) throws  Exception{

        DataFrame trainingDataFrame  = getTrainingDataFrame(jsc, mlapi, datasetId);
        List<String> columnsToBeAssembled = new LinkedList<String>();
        for (String column : trainingDataFrame.columns()) {
            columnsToBeAssembled.add(column);
        }
        //remove label column if exists
        columnsToBeAssembled.remove("label");
        columnsToBeAssembled.remove("accountId");

        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(columnsToBeAssembled.toArray(ArrayUtils.EMPTY_STRING_ARRAY));
        vectorAssembler.setOutputCol("features");

        Log1PScaler logScaler = new Log1PScaler();
        logScaler.setInputCol("features");
        logScaler.setOutputCol("scaledFeatures");

        LogisticRegression logisticRegression = new LogisticRegression();
        logisticRegression.setRegParam(0.01);
        logisticRegression.setMaxIter(100);
        logisticRegression.setFeaturesCol("scaledFeatures");
        logisticRegression.setLabelCol("label");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{vectorAssembler, logScaler, logisticRegression});

        PipelineModel pipelineModel = pipeline.fit(trainingDataFrame);
        MLAPI modelApi = MLAPI.getInstance();
        ModelId modelId = publishModel(modelApi, pipelineModel, trainingDataFrame);
        System.out.println(modelId.getId());
        System.out.println(modelId.getVersion());
        return pipelineModel;
    }

    public static DataFrame getTrainingDataFrame(JavaSparkContext jsc, SparkMLAPI mlapi, DatasetId datasetId)
            throws MLAPIClientErrorException, MLAPISDKException{
        DataFrame trainingDataFrame = mlapi.loadDataset(datasetId, jsc.sc());
        trainingDataFrame = trainingDataFrame.withColumn("labelTmp", trainingDataFrame.col("label").cast(DoubleType))
                .drop("label")
                .withColumnRenamed("labelTmp", "label");
//        trainingDataFrame.select("label").show();
        return trainingDataFrame;
    }

    public static ModelId publishModel(MLAPI mlapi, Model model, DataFrame dataFrame) {
        dataFrame.show();
        final byte[] serializedModel = ModelExporter.export(model, dataFrame);
        Transformer transformer = ModelImporter.importAndGetTransformer(serializedModel);
        final ModelMetaData modelMetadata = prepareModelMetaData(transformer, dataFrame);

        try {
            return mlapi.publishModel(serializedModel, modelMetadata);
        } catch (MLAPIException e) {
            throw new RuntimeException("Exception in publishing model", e);
        }

    }

    public static ModelMetaData prepareModelMetaData(Transformer transformer, DataFrame trainingDf) {

        //Get key value pair of input with their types
        String jsonSerializedInputs = SchemaExporter.exportToJson( transformer.getInputKeys(), trainingDf.schema() );

        //Deserialize it into model sdk format
        List<Input> inputsList = gson.fromJson(jsonSerializedInputs, new TypeToken<ArrayList<Input>>(){}.getType());
        for(Input inp: inputsList){
            System.out.println(inp.getDatatype() + " " + inp.getName().toString() + " "+ inp.getId());
        }
        List<Output> outputsList = new ArrayList<>();
        //cannot extract output data types from training df :(
        for(String key : transformer.getOutputKeys()) {
            Output output = new Output();
            output.setName(key);
            outputsList.add(output);
        }
        ModelMetaData modelMetadata = new ModelMetaData();
        modelMetadata.setOwner("ml-platform@flipkart.com, karan.verma@flipkart.com");
        modelMetadata.setNamespace("UIE");
        modelMetadata.setModelName("GenderDemographic");
        modelMetadata.setDescription("GenderClassification");
        modelMetadata.setModelState(ModelState.STAGE);
        modelMetadata.setInputs(inputsList);
        modelMetadata.setOutputs(outputsList);
        return modelMetadata;
    }

    public static void predLabelFile(Model model, DatasetId datasetId, SparkMLAPI mlapi, JavaSparkContext jsc, String hdfsPath, String fileName) throws Exception{
        DataFrame testDataFrame  = getTrainingDataFrame(jsc, mlapi, datasetId);
        DataFrame withPredictions = model.transform(testDataFrame);
        withPredictions.select("accountId", "probability", "label").toJSON().saveAsTextFile(hdfsPath + fileName);
    }






}

