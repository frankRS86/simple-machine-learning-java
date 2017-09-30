package ml.base;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.ByteBuffer;


/**
 * Copyright Frank Siller 2017
 */
public class Computations
{
	public static double sigmoid(double z)
	{
		double g = 1.0 / (1.0 + Math.exp(-z));
		return g;
	}
	
	public static double[] sigmoid(double[] v) 
	{
		double[] C = new double[v.length];
		for(int i = 0; i < v.length;i++)
		{
			C[i] = 1.0 / (Math.exp(-v[i]) + 1.0);
		}
		
		return C;
	}
	
	public static int maxIndex(double[] v)
	{
		double max = 0.0;
		int idx = 0;
		for (int i = 0; i < v.length;i++)
		{
			if(v[i] >= max)
			{
				max = v[i];
				idx = i;
			}			
		}
		return idx;
		
	}

	public static double sigmoidGradient(double z)
	{
		double g = sigmoid(z);
		double sg = g * (1 - g);
		return sg;
	}
	
	public static double[] sigmoidGradient(double[] ds) {
		
		double[] C = new double[ds.length];
		
		for(int i = 0; i < ds.length;i++)
		{
			double v = sigmoid(ds[i]);
			C[i] = v * (1-v);
		}
		
		return C;
	}

	public static double round(double value, int places)
	{
		if (places < 0)
			throw new IllegalArgumentException();

		BigDecimal bd = new BigDecimal(value);
		bd = bd.setScale(places, RoundingMode.HALF_UP);
		return bd.doubleValue();
	}

	public static double[] toDoubleArray(byte[] byteArray)
	{
		int times = Double.SIZE / Byte.SIZE;
		double[] doubles = new double[byteArray.length / times];
		for (int i = 0; i < doubles.length; i++)
		{
			doubles[i] = ByteBuffer.wrap(byteArray, i * times, times).getDouble();
		}
		return doubles;
	}

	public static double[][] sigmoid(double[][] ds) {
		
		double [][]C = new double[ds.length][ds[0].length];
		
		for (int i = 0;i<ds.length;i++)
		{
			for(int j = 0; j < ds[0].length;j++)
			{
				C[i][j] = 1.0 / (Math.exp(-ds[i][j]) + 1.0);
			}
		}
		return C;
	}




}
