package la;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import ml.base.Computations;

/**
 * 
 * @author Frank Siller (c) 2017
 *
 */
public class MatrixOperations 
{
	public static FollowupOperation POW2 = (val,i,j) ->
	{
		return Math.pow(val,2);
	};
	
	public static FollowupOperation LOG = (val,i,j) ->
	{
		return Math.log(val);
	};
	
	public static FollowupOperation SQRT = (val,i,j) ->
	{
		return Math.sqrt(val);
	};
	
	public static FollowupOperation SIGMOID = (val,i,j) ->
	{
		return Computations.sigmoid(val);
	};
	
	public static FollowupOperation SUBTRACTFROM1 = (val,i,j) ->
	{
		return 1 - val;
	};
	
	/**
	 * Splits the data in threadCount sets. MatrixOperation op is executed on each subset in a separate thread
	 * @param A
	 * @param op
	 * @param threadCount number of splits 
	 */
	private static void executeMultithreaded(double[][]A,MatrixOperation op, int threadCount)
	{
		int[][] border = new int[threadCount][2];
		int m = A.length;
		
		if(A.length == 1)
		{
			threadCount = 1;
		}
		
		for(int i = 0;i < threadCount; i++)
		{
			int partSize = m/threadCount;
			border[i] = new int[]{i*partSize,(i+1)*partSize};
		}
		
		List<Thread> threads = new ArrayList<Thread>();
		
		for(int t = 0; t < threadCount; t++)
		{
			final int step = t;
			
			Runnable task = () -> 
			{
				op.compute(border[step][0], border[step][1]);
			};
			
			Thread tx = new Thread(task);
			threads.add(tx);
			
			tx.start();
		}
		
		try 
		{
			for (Thread thread : threads) {
				  thread.join();
				}
		
		} catch (InterruptedException e) {
			
			e.printStackTrace();
		}	
	}
	
	/**
	 * 
	 * @param A matrix of dimension m x n
	 * @param B matrix of dimension n x k
	 * @param threadCount number of threads the calculation is executed on. Waits till all threads finish
	 * @return matrix of dimension m x k (n of inputs must be equal)
	 */
	public static double[][] mult(double[][]A, double[][] B, int threadCount,Optional<FollowupOperation> follow)
	{		
			int colB = B[0].length;
			int n = B.length;
			double[][] C = new double[A.length][colB];
		
			MatrixOperation op = (int start, int end) -> 
			{
				for(int i = start; i < end;i++)
				{
					for (int k = 0; k  < colB; k++)
					{
						for (int j = 0; j < n;j++)
						{
							C[i][k] += A[i][j] * B[j][k]; 
						}
						C[i][k] = follow.orElse( (val,r,c) -> val).execute(C[i][k],i,k);
					}
				}
			};
		
		executeMultithreaded(A,op, threadCount);
		return C;
	}
	
	/**
	 * @param A matrix of dimension m x n
	 * @param v column vector of length n
	 * @param threadCount
	 * @return column vector of dimension m
	 */
	public static double[] multMxCV(double[][]A, double[] v, int threadCount,Optional<FollowupOperation> follow)
	{
		double[][]colMatrix = new double[v.length][1];
		
		for(int i = 0; i < v.length;i++)
		{
			colMatrix[i][0] = v[i];
		}
		
		double[][] C = mult(A, colMatrix, threadCount,follow);
		double[] vector = new double[C.length];
		
		int n = C.length;
		for(int i = 0; i < n;i++)
		{
			vector[i] = C[i][0];
		}
	
		return vector;	
	}
	
	/**
	 * @param v row vector of length m
	 * @param A matrix of dimension m x n
	 * @param threadCount
	 * @return row "vector" of length n (1xn Matrix)
	 */
	public static double[][] multRVxM(double[] v,double[][]A, int threadCount,Optional<FollowupOperation>follow)
	{
		double[][]rowMatrix = new double[1][v.length];
		rowMatrix[0] = v;

		return mult(rowMatrix,A, threadCount,follow);
	}
	
	/**
	 * 
	 * @param a a column vector of length m 
	 * @param b a row vector of length n 
	 * @param threadCount
	 * @param followUp
	 * @return matrix of dimension m x n
	 */
	public static double[][] multCVxRV(double[]a, double[] b, int threadCount,Optional<FollowupOperation> followUp)
	{
		double[][]colMatrix = new double[a.length][1];
		double[][]rowMatrix = new double[1][b.length];
		
		rowMatrix[0] = b;
		
		for(int i = 0; i < a.length;i++)
		{
			colMatrix[i][0] = a[i];
		}
		
		return mult(colMatrix,rowMatrix, threadCount,followUp);
	}
	
	/**
	 * 
	 * @param a rowVector of length n
	 * @param b column vector of length n
	 * @param threadCount
	 * @return returns a 1x1 Matrix
	 */
	public static double[][] multRVxCV(double[]a, double[] b, int threadCount)
	{
		double[][]rowMatrix = new double[1][a.length];
		double[][]colMatrix = new double[b.length][1];
	
		rowMatrix[0] = a;
		
		for(int i = 0; i < b.length;i++)
		{
			colMatrix[i][0] = b[i];
		}
		
		return mult(rowMatrix,colMatrix, threadCount,Optional.empty());
	}
	
	
	/**
	 * splits a matrix a by given row number splitIndex in two matrices
	 * the columns stay untouched
	 * @param a matrix of dimension m x n
	 * @param splitIndex the row index to split
	 * @return array of two matrices of dimensions splitIndex x n and splitindex+1 x n
	 */
	public static double[][][] split(double[][] a,int splitIndex) 
	{
		int firstTo = splitIndex;
		int secondFrom = splitIndex+1;
		int secondSize = a.length - secondFrom;
		
		double [][][]split = new double[2][0][0];
		split[0] = new double[firstTo+1][a[0].length];
		split[1] = new double[secondSize][a[0].length];
		
		System.arraycopy(a, 0, split[0], 0, firstTo+1);
		System.arraycopy(a, secondFrom, split[1], 0, (secondSize));
		
		return split;
	}


	/**
	 * 
	 * @param A matrix of dimension m x n
	 * @param B matrix of dimension n x k
	 * @return matrix of dimension m x k (n of inputs must be equal)
	 */
	public static double[][] multSingleThread(double[][] A, double[][] B) {
		
		int m = A.length;
		int n = B.length;
		int colB = B[0].length;
		
		double[][]C = new double[m][colB];

		for(int i = 0; i < m; i++)
		{
			for (int k = 0; k  < colB; k++)
			{
				for (int j = 0; j < n;j++)
				{
					C[i][k] += A[i][j] * B[j][k]; 
				}
			}
		}
		
		return C;
	}
	
	/**
	 * inserts a new column at position 0 in matrix A. each element will have value val.
	 * 
	 * @param A matrix of dimension m x n
	 * @param val
	 * @return new Matrix of dimension m x n+1
	 */
	public static double [][] insertConstFirstCol(double[][] A,double val)
	{
		double[][]C = new double[A.length][A[0].length +1];
		
		MatrixOperation op = (int start, int end) ->
		{
			int n = A[0].length;
			
			for(int i = start; i < end; i++)
			{
				C[i][0] = val;
				for(int j = 1; j < n;j++)
				{
					C[i][j] = A[i][j-1];
				}
			}
		};
		executeMultithreaded(A, op, 2);
		return C;
		
	}
	
	/**
	 * generates from A(m x n) -> C(n x m). Switches rows and columns
	 * @param A
	 * @return C[j][i] = A[i][j]
	 */
	public static double[][] transpose(double [][] A)
	{
		int m = A.length;
		int n = A[0].length;
		
        double[][] C = new double[n][m];
        
        MatrixOperation op = (int start,int end) -> {
        
        for (int i = start; i < end; i++)
        {
            for (int j = 0; j < n; j++)
            {
                C[j][i] = A[i][j];
            }
        }
        
        };
        
        executeMultithreaded(A, op, 2);
        return C;
    }
	
	/**
	 * executes any operation on matrix A. Its executed in numThreads Threads.
	 * the given Matrix operation is called numThreads times with start and end parameters
	 * for the part the specific thread is working on.
	 * @param A
	 * @param op
	 */
	public void executeCustomOp(double[][] A,MatrixOperation op,int numThreads)
	{
		//double[][] C= new double [A.length][A[0].length];
		executeMultithreaded(A, op, numThreads);
	}

	/**
	 * subtracts matrix B from Matrix A. returns a new matrix with the subtraction results
	 * @param A
	 * @param B
	 * @return
	 */
	public static double[][] subtract(double[][] A, double[][] B,Optional<FollowupOperation>follow) {
		
		int n = A[0].length;
		double C[][] = new double[A.length][n]; 
		
		MatrixOperation op = (start,end) -> 
		{
			for(int i = start;i < end;i++)
			{
				for(int j = 0; j < n;j++)
				{
					C[i][j] = A[i][j] - B[i][j];
					C[i][j] = follow.orElse( (val,r,c) -> val).execute(C[i][j],i,j);
				}
			}
			
		};
		executeMultithreaded(A, op, 2);
		
		return C;
	}
	
	/**
	 * add two matrices. returns a new matrix with the addition results
	 * @param A
	 * @param B
	 * @return
	 */
	public static double[][] add(double[][] A, double[][] B) {
		
		int n = A[0].length;
		double C[][] = new double[A.length][n]; 
		
		MatrixOperation op = (start,end) -> 
		{
			for(int i = start;i < end;i++)
			{
				for(int j = 0; j < n;j++)
				{
					C[i][j] = A[i][j] + B[i][j];
				}
			}
			
		};
		executeMultithreaded(A, op, 2);
		
		return C;
	}
	
	/**
	 * Adds two matrices and sums up the result matrix
	 * @param A
	 * @param B
	 * @return
	 */
	public static double addAndSum(double[][] A, double[][] B) {
		
		int n = A[0].length;
		double C[][] = new double[A.length][n]; 
		final List<Double> sums = new ArrayList<Double>();
		
		MatrixOperation op = (start,end) -> 
		{
			double partialSum = 0;
			
			for(int i = start;i < end;i++)
			{
				for(int j = 0; j < n;j++)
				{
					C[i][j] = A[i][j] + B[i][j];
					partialSum+=C[i][j];
				}
			}
			sums.add(partialSum);
			
		};
		executeMultithreaded(A, op, 2);
		
		double sum = 0;
		for(double val : sums)
			sum+=val;
		
		return sum;
	}



	/**
	 * Removes column col from matrix A. executes follow up operation on result
	 * @param A
	 * @param col
	 * @return
	 */
	public static double[][] slice(double[][] A,int col,Optional<FollowupOperation>follow) 
	{		
		double[][] C = new double[A.length][A[0].length -1];
		int colLength = C[0].length;
		if(colLength == 0)
		{
			return C;
		}
		
		MatrixOperation op = (start,end) ->
		{
			for(int i = start ; i < end;i++)
			{
					System.arraycopy(A[i],0,C[i],0,col);
					System.arraycopy(A[i], col+1,C[i], col, A[i].length-(col+1));
					
					for(int j = 0; j < colLength;j++)
					{
						C[i][j] = follow.orElse( (val,r,c) -> val).execute(C[i][j],i,j);
					}
			}
		};
		executeMultithreaded(A, op, 2);
		return C;
	}
	
	/**
	 * Removes column col from matrix A and transposes the result matrix in one step
	 * @param A
	 * @param col
	 * @return
	 */
	public static double[][] sliceAndTranspose(double[][] A,int col,Optional<FollowupOperation>follow) 
	{		
		double[][] C = new double[A[0].length -1][A.length];
		int rowLength = C.length;
		if(rowLength == 0)
		{
			return C;
		}
		
		MatrixOperation op = (start,end) ->
		{
			for(int i = start ; i < end;i++)
			{
					double []tmp = new double[rowLength];
					System.arraycopy(A[i],0,tmp,0,col);
					System.arraycopy(A[i], col+1,tmp, col, A[i].length-(col+1));
					
					for(int row = 0; row < rowLength;row++)
					{
						C[row][i] = tmp[row];
						C[row][i] = follow.orElse( (val,r,c) -> val).execute(C[row][i],row,i);
					}
			}
		};
		executeMultithreaded(A, op, 2);
		return C;
	}
	
	/**
	 * Removes column col from matrix A and sums up all left elements
	 * @param A
	 * @param col
	 * @return
	 */
	public static double sliceandSum(double[][] A,int col,Optional<FollowupOperation>follow) 
	{		
		double[][] C = new double[A.length][A[0].length -1];
		double sum = 0;
		int m = C.length;
		int colLength = C[0].length;
		if(colLength == 0)
		{
			return 0;
		}
		
		for(int i = 0 ; i < m;i++)
		{
			System.arraycopy(A[i],0,C[i],0,col);	
			System.arraycopy(A[i], col+1,C[i], col, A[i].length-(col+1));
							
			for(int j = 0; j < colLength;j++)
			{
				C[i][j] = follow.orElse( (val,r,c) -> val).execute(C[i][j],i,j);
				sum+=C[i][j];
			}
		}
		
		return sum;
	}

	/**
	 * multiplicates each value of matrix A with value d
	 * @param A
	 * @param d
	 * @return
	 */
	public static double[][] mult(double[][] A, double d) {
		
		double [][]C = new double[A.length][A[0].length];
		int n = A[0].length;
		
		MatrixOperation op = (start,end) ->
		{
			for(int i = start; i < end;i++)
			{
				for(int j = 0;j<n;j++)
				{
					C[i][j] = A[i][j] * d; 
				}
		
			}
		};
		
		executeMultithreaded(A, op, 2);
		return A;
		
	}

	/**
	 * returns column col of matrix A as a vector
	 * @param A
	 * @param col
	 * @return
	 */
	public static double[] getColumn(double[][] A, int col) 
	{
		double []v = new double[A.length];
		
		for(int i = 0; i < A.length;i++)
		{
			v[i] = A[i][col];
		}
		
		return v;
	}

	/**
	 * replaces all values of column idx with corresponding vector col 
	 * @param A
	 * @param col
	 * @param idx
	 */
	public static void setColumn(double[][] A, double[] col, int idx) 
	{
		int m = A.length;
				
		for(int i = 0; i < m; i++)
		{
			A[i][idx] = col[i];
		}
	}
	
	
	/**
	 * sets all elements of column col of Matrix A to value val
	 * @param A
	 * @param col
	 * @param val
	 * @return
	 */
	public static double[][] replaceColumnWithValue(double[][] A, int col, int val) {
		
		MatrixOperation op = (start,end) ->
		{
			for(int i = start ; i < end;i++)
			{
				A[i][col] = val; 
			}
		};
		
		executeMultithreaded(A, op, 2);
		return A;
		
	}
		
	/**
	 * executes all given operations ops on each element of matrix A sequential.
	 * so e.g. A[1][1] = 2, (A,MatrixOperations.POW2,(v,i,j)-> 1 - v)
	 * would result in A[1][1] = -3.
	 * @param A
	 * @param ops
	 * @return a new matrix of size A with executed results
	 */
	public static double[][] executeOnEachElement(double[][] A,FollowupOperation ...ops)
	{
		int n = A[0].length;
		final double[][] C = new double[A.length][n];
		
		MatrixOperation op = (start,stop) ->{
		
		for(int i = start; i < stop;i++)
		{
			for (int j = 0; j < n;j++)
			{
				C[i][j] = ops[0].execute(A[i][j],i,j); 
				//System.out.println(i+","+j+": "+C[i][j]);
				
				for(int o = 1; o < ops.length;o++)
				{
					C[i][j] = ops[o].execute(C[i][j],i,j); 
					//System.out.println(i+","+j+": "+C[i][j]);
				}
			}
		}
		};
		executeMultithreaded(A, op, 2);
		return C;
	}
	
	/**
	 * executes all given operations ops on each element of matrix A sequential.
	 * Sums up all results.
	 * so e.g. A[1][1] = 2, (A,MatrixOperations.POW2,(v,i,j)-> 1 - v)
	 * would result in A[1][1] = -3.
	 * @param A
	 * @param ops
	 * @return the sum of all elements;
	 */
	public static double executeOnEachEementAndSumUp(double[][] A,FollowupOperation ...ops)
	{
		int n = A[0].length;
		int m = A.length;
		double[][] C = new double[m][n];
		double val = 0;
				
		for(int i = 0; i < m;i++)
		{
			for (int j = 0; j < n;j++)
			{
				C[i][j] = ops[0].execute(A[i][j],i,j); 
				
				for(int o = 1; o < ops.length;o++)
				{
					C[i][j] = ops[o].execute(C[i][j],i,j); 
				}
				
				val+=C[i][j];
			}
		}
		return val;
	}


	
	
	
}
