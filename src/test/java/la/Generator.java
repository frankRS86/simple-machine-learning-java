package la;

import java.util.Random;

/**
 * 
 * @author Frank Siller (c) 2017
 *
 */
public class Generator {

	public static double[][] create(int m,int n)
	{
		Random rand = new Random();
		double [][] A = new double[m][n];
		for(int i = 0; i < A.length;i++)
		{
			for(int j = 0; j < n;j++)
			{
				A[i][j] = rand.nextDouble() * 0.12 - 0.12;
			}
		}
		
		return A;
	}

	public static double[][] createWithConstant(int m, int n) 
	{
		double [][] A = new double[m][n];
		for(int i = 0; i < A.length;i++)
		{
			for(int j = 0; j < n;j++)
			{
				A[i][j] = (double)i;
			}
		}
		
		return A;
	}

	public static double[] createRow(int num) 
	{
		double [] A = new double[num];
		for(int i = 0; i < A.length;i++)
		{
			A[i] = i;
		}
		
		return A;
	}
}
