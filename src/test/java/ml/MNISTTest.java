package ml;

import org.junit.Test;

import ml.base.Computations;
import ml.base.FeatureSet;
import ml.base.MNISTData;
import ml.gauss.GaussianNormal;
import ml.neuralnet.NeuralNet;

public class MNISTTest 
{
	@Test
	public void testMNISTData()
	{
		
		MNISTData data = new MNISTData("train-labels-idx1-ubyte", "train-images-idx3-ubyte",12);
		FeatureSet set = data.getData();
			
		//needed?
		set.normalise();
		
		double bestCost = 100;
		double bestHits = 0;
		double bestAlpha = 0;
		double bestLayerSize = 0;
		
		for(int layerSize = 80;layerSize < 200;layerSize+=10)
		for (double alpha = 0.01; alpha < 1;alpha+=0.03)
		{
		
		System.out.println("training nn alpha: "+alpha+ "hiddenlayer size: "+layerSize);
		NeuralNet nn = new NeuralNet(784, layerSize, 80, 10);
		nn.init();
		double finalCost = nn.train(set, 200,alpha,0);
		
		System.out.println("training finished");

		MNISTData test = new MNISTData("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte",2);
		FeatureSet testImages = test.getData();
		double[][] labelsTest = testImages.getLabelMatrix();
		double[][] featuresTest = testImages.features;

		int hits = 0;
		for (int i = 0; i < featuresTest.length; i++)
		{
			double [] x = featuresTest[i];
			double [] y = labelsTest[i];
			
			double [] h = nn.predict(x);
			
			int exp = Computations.maxIndex(y);
			//System.out.println("img:"+exp);
			int res = Computations.maxIndex(h);
			//System.out.println("pred:"+res);
			
			if(exp == res)
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
		
		
		System.out.println(hits+" hits!! = "+percent+"%");
		System.out.println("best hits: "+bestHits+"% best cost: "+bestCost+" a: "+bestAlpha+" layer: "+bestLayerSize);
		System.out.println("-------------------------------------------------------------");
		
		}
	}
}
