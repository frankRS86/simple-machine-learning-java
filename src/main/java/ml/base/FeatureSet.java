package ml.base;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import com.sun.xml.internal.fastinfoset.sax.Features;

import la.MatrixOperations;

public class FeatureSet
{
	List<String> exampleIds;

	public List<double[]> tmpVals = new ArrayList();
	
	public List<List<Double>> listVals = new ArrayList();
	
	public double[][] features;
	
	public double[][] labels;

	public FeatureSet()
	{
		exampleIds = new ArrayList<String>();
	}

	/**
	 * normalizes each feature to range -1 to 1
	 * 
	 * x = (x - mean) / max
	 */
	public void normalise()
	{
		long start = System.currentTimeMillis();

		int m = features.length;
		
		for (int i = 0; i < features[0].length; i++)
		{
			double []col = MatrixOperations.getColumn(features,i);
			
			double max = Double.MIN_VALUE;
			double sum = 0.0;
			for(int r = 0; r < col.length; r++)
			{
				if(col[r] > max)
				{
					max = col[r];
				}
				
				sum+=col[r];
			}
			double mean = sum / (double)m;

			for(int r = 0; r < col.length;r++)
			{
				col[r] = (col[r] - mean) / max;
			}

			MatrixOperations.setColumn(features,col,i);
		}

		long end = System.currentTimeMillis();
		System.out.println("normalization took ms:" + (end - start));
	}

	public double[][] getFeatures()
	{
		return features;
	}

	public String getFeatureExampleName(int index)
	{
		if (index >= exampleIds.size())
		{
			return "INVALID";
		}
		return exampleIds.get(index);
	}

	public int getExampleSize()
	{
		return features.length;
	}

	public double[] getExample(int idx)
	{
		return features[idx];
	}

	public double[][] getLabelMatrix()
	{
		return labels;
	}

	public void shuffle() 
	{
		// If running on Java 6 or older, use `new Random()` on RHS here
	    Random rnd = ThreadLocalRandom.current();
	    for (int i = features.length - 1; i > 0; i--)
	    {
	      int index = rnd.nextInt(i + 1);
	      double[] row = features[index];
	      features[index] = features[i];
	      features[i] = row;
	    }
	}
	public void setLabelMatrix(double[][] d) 
	{
		labels = d;		
	}

	public void setFeatureMatrix(double[][] d) 
	{
		features = d;		
	}

	
	public double[][] getFeatures(int from,int to)
	{
		int size = to - from;
		double[][] f = new double[size][features[0].length];
		System.arraycopy(features,from, f, 0, size);
		return f;
	}
	
	public double[][] getLabels(int from,int to)
	{
		int size = to - from;
		double[][] f = new double[size][labels[0].length];
		System.arraycopy(labels,from, f, 0, size);
		return f;
	}
	
	public FeatureSet[] split() 
	{
		double[][][] fsplit = MatrixOperations.split(features, 2);
		double[][][] lsplit = MatrixOperations.split(labels, 2);
		FeatureSet[] subsets = new FeatureSet[2];
		subsets[0] = new FeatureSet();
		subsets[0].features = fsplit[0];
		subsets[0].labels = lsplit[0];
		
		subsets[1] = new FeatureSet();
		subsets[1].features = fsplit[1];
		subsets[1].labels = lsplit[1];
		
		return subsets;
	}


	public double[] getLabel(int i) {
		
		return labels[i];
	}
}
