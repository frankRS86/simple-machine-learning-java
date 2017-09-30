package la;

import java.util.Optional;

public class VectorOperations 
{
	/**
	 * generates a new vector of size length filled with 0
	 * @param length
	 * @return
	 */
	public static double[] zeros(int length)
	{
		return constant(length, 0.0);
	}
	
	/**
	 * generates a new vector of size length filled with 1
	 * @param length
	 * @return
	 */
	public static double[] ones(int length)
	{
		return constant(length, 1.0);
	}
	
	/**
	 * generates a new vector of size length filled with val
	 * @param length
	 * @return
	 */
	public static double[] constant(int length,double val)
	{
		double[] v = new double[length];
		for(int i = 0; i < length;i++)
		{
			v[i] = val;
		}
		return v;
	}

	public static double[] subtract(double[] v, double d,FollowupOperation ...follow) 
	{
		int n = v.length;
		double[] c = new double[n];
		for (int i = 0; i < n;i++)
		{
			c[i] = v[i] - d;
			if(follow.length == 0)
			{
				continue;
			}
			for(int o = 0; o < follow.length; o++)
			c[i] = follow[o].execute(c[i], i, -1);
		}
		return c;
	}
	
	public static double[] subtract(double[] v, double[] b,FollowupOperation ...follow) 
	{
		int n = v.length;
		double[] c = new double[n];
		for (int i = 0; i < n;i++)
		{
			c[i] = v[i] - b[i];
			if(follow.length == 0)
			{
				continue;
			}
			for(int o = 0; o < follow.length; o++)
			c[i] = follow[o].execute(c[i], i, -1);
		}
		return c;
	}
	
	/**
	 * removes the first element of a vector
	 * @param ds
	 * @return
	 */
	public static double[] slice(double[] ds) {
		
		double[] r = new double[ds.length - 1];
		System.arraycopy(ds, 1, r, 0, ds.length-1);
		
		return r;
	}
	
	/**
	 * inserts value val at position 0 in vector v. all other values are shifted right
	 * @param v vector of length n
	 * @param val
	 * @return vector of length n+1
	 */
	public static double[] insertConstFirst(double[] v, double val) 
	{
		int n = v.length +1;
		double []c = new double[n];			
		c[0] = val;
		
		System.arraycopy(v, 0, c, 1, v.length);
		return c;
	}
	
	/**
	 * add two vectors a and b
	 * @param v vector of length n
	 * @param val
	 * @return vector of length n
	 */
	public static double[] add(double[] a, double []b) 
	{
		int n = a.length;
		double []c = new double[n];			
		
		for (int i = 0; i < n;i++)
		{
			c[i] = a[i] - b[i];
		}
		
		return c;
	}

	public static double[] divide(double[] a, double val) {
		int n = a.length;
		double []c = new double[n];			
		
		for (int i = 0; i < n;i++)
		{
			c[i] = a[i] / val;
		}
		
		return c;
	}
	
	public static double sum(double[] a)
	{
		int n = a.length;
		double sum = 0;		
		
		for (int i = 0; i < n;i++)
		{
			sum+=a[i];
		}
		return sum;
	}

	public static double product(double[] a) {
		int n = a.length;
		double sum = 1;		
		
		for (int i = 0; i < n;i++)
		{
			sum*=a[i];
		}
		return sum;
	}
}
