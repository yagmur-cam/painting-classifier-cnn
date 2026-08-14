package org.example;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.datavec.image.transform.*;
import org.datavec.api.io.labels.ParentPathLabelGenerator; 
import org.datavec.api.split.FileSplit; 
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;


public class DataLoader {
    public static final int IMAGE_HEIGHT = 128;
    public static final int IMAGE_WIDTH = 128;
    public static final int CHANNELS = 3;
    public static final int NUM_CLASSES = 7;
    public static final int BATCH_SIZE = 32;

    private String datasetPath;

    public DataLoader(String datasetPath) { //constructor for the DataLoader
        this.datasetPath = datasetPath;
    }

    public String getDatasetPath() {
        return datasetPath;
    }
    public DataSetIterator loadData(String splitFolder) throws IOException {
        File folder = new File(datasetPath + "/" + splitFolder);
        FileSplit fileSplit = new FileSplit(folder, NativeImageLoader.ALLOWED_FORMATS, new Random(42)); //general randomness
        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, labelGenerator);
        if (splitFolder.equals("train")) {
            ImageTransform flipTransform = new FlipImageTransform(1);
            ImageTransform rotateTransform = new RotateImageTransform(15);
            ImageTransform cropTransform = new RandomCropTransform(200, 200);
            List<Pair<ImageTransform, Double>> transforms = new ArrayList<>();
            transforms.add(new Pair<>(flipTransform, 0.5));
            transforms.add(new Pair<>(rotateTransform, 0.5));
            transforms.add(new Pair<>(cropTransform, 0.5));
            ImageTransform pipeline = new PipelineImageTransform(transforms);
            recordReader.initialize(fileSplit, pipeline);
        } else {
            recordReader.initialize(fileSplit);
        }
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, BATCH_SIZE, 1, NUM_CLASSES);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(iterator);
        iterator.setPreProcessor(scaler);
        return iterator;
    }
}
