package ml;

import org.junit.Test;

import ml.base.Computations;
import ml.base.FeatureSet;
import ml.base.MNISTData;
import ml.gauss.GaussianNormal;
import ml.neuralnet.NeuralNet;
import ml.neuralnet.NeuralNet.CostResult;

public class MNISTTest 
{
	@Test
	public void testMNISTData()
	{
		int trainingSize = 10;
		
		for(int t = 100; t < 10000;t+=50)
		{
		
		MNISTData data = new MNISTData("train-labels-idx1-ubyte", "train-images-idx3-ubyte",t);
		FeatureSet set = data.getData();
			
		//needed?
		set.normalise();
		
		double bestCost = 100;
		double bestHits = 0;
		double bestAlpha = 0;
		double bestLayerSize = 0;
		
		int layerSize= 120;
		int secondLayer = 50;
		int thirdLayer = 100;
		double alpha = 0.01;
		double lambda = 20;
		
		
		//for(int layerSize = 200;layerSize < 400;layerSize+=10)
		//for (double alpha = 0.01; alpha < 1;alpha+=0.03)
		{
		
		System.out.println("training nn alpha: "+alpha+ " hiddenlayer size: "+layerSize+" second layer"+secondLayer);
		NeuralNet nn = new NeuralNet(784, layerSize, secondLayer, 10);
		nn.init();
		double finalCost = nn.train(set, 500,alpha,lambda);
		
		System.out.println("training finished");

		MNISTData test = new MNISTData("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte",10000);
		FeatureSet testImages = test.getData();
		double[][] labelsTest = testImages.getLabelMatrix();
		double[][] featuresTest = testImages.features;

		CostResult res = nn.calculateCost(featuresTest, labelsTest, lambda);
		
		int hits = 0;
		for (int i = 0; i < res.h.length; i++)
		{
			double [] h = res.h[i];
			double [] y = labelsTest[i];
			
			int exp = Computations.maxIndex(y);
			//System.out.println("img:"+exp);
			int pred = Computations.maxIndex(h);
			//System.out.println("pred:"+res);
			
			if(exp == pred)
			{
				hits++;
			}
		}
		
		double percent = (((double)hits)/(double)featuresTest.length)*100.0;
		
		if(percent > bestHits)
		{
			bestHits = percent;
		}
		
		if(finalCost < bestCost)
		{
			bestCost = finalCost;
			bestAlpha = alpha;
			bestLayerSize = layerSize;
		}
		
		
		System.out.println(hits+" hits!! = "+percent+"% at triningsize: "+t);
		System.out.println("best hits: "+bestHits+"% alpha: "+bestAlpha+" layer: "+bestLayerSize);
		System.out.println("m = "+t+" J train: "+bestCost);
		System.out.println("m = "+t+" J test: "+res.J);
		System.out.println("-------------------------------------------------------------");
		
		}
	}
	}
}
