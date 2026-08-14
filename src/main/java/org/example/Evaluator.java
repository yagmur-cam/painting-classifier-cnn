package org.example;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;

public class Evaluator {
    public void evaluate(String datasetPath) throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("painting-classifier-model.zip"), true);
        DataLoader data = new DataLoader(datasetPath);
        DataSetIterator testIterator = data.loadData("test");
        Evaluation testEval = model.evaluate(testIterator);
        System.out.println(testEval.stats());
    }

    public String predict(String imagePath) throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("painting-classifier-model.zip"), true);
        NativeImageLoader loader = new NativeImageLoader(128, 128, 3);
        INDArray image = loader.asMatrix(new File(imagePath));
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);
        INDArray output = model.output(image);
        int predictedClass = output.argMax(1).getInt(0);
        String[] styles = {"Baroque", "Cubism", "Expressionism", "Impressionism", "Renaissance", "Romanticism", "Surrealism"};
        return styles[predictedClass];
    }
    public double[] predictProbabilities(String imagePath) throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("painting-classifier-model.zip"), true);
        NativeImageLoader loader = new NativeImageLoader(128, 128, 3);
        INDArray image = loader.asMatrix(new File(imagePath));
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);
        INDArray output = model.output(image);
        double[] probs = new double[7];
        for (int i = 0; i < 7; i++) {
            probs[i] = output.getDouble(i);
        }
        return probs;
    }
}
