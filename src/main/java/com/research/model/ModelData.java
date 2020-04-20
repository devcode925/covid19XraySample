package com.research.model;

public class ModelData {
    private int channels=0;
    private int height=0;
    private int width=0;
    private int numberOfLabels=0;
    private long seed=24;

    public ModelData( int channels, int height, int width,int numberOfLabels, long seed) {
        this.channels=channels;
        this.height=height;
        this.width=width;
        this.numberOfLabels=numberOfLabels;
        this.seed=seed;
    }
    public int getChannels() {
        return channels;
    }

    public void setChannels(int channels) {
        this.channels = channels;
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public int getNumberOfLabels() {
        return numberOfLabels;
    }

    public void setNumberOfLabels(int numberOfLabels) {
        this.numberOfLabels = numberOfLabels;
    }

    public long getSeed() {
        return seed;
    }

    public void setSeed(long seed) {
        this.seed = seed;
    }
}
