package la;

import java.util.Optional;

import org.junit.Assert;
import org.junit.Test;


/**
 * 
 * @author Frank Siller (c) 2017
 *
 */
public class MultTest {

	@Test
	public void testSmall()
	{
		double [][] A = new double[2][3];
		A[0] = new double[]{3,2,1};
		A[1] = new double[]{1,0,2};
		
		double [][] B = new double[3][2];
		B[0] = new double[]{1,2};
		B[1] = new double[]{0,1};
		B[2] = new double[]{4,0};
		
		
		double [][]res = MatrixOperations.mult(A, B,2,Optional.empty());
		
		Assert.assertEquals(2, res.length ,0);
		Assert.assertEquals(2, res[0].length,0);
		
		Assert.assertEquals(7, res[0][0],0);
		Assert.assertEquals(8, res[0][1],0);
		Assert.assertEquals(9, res[1][0],0);
		Assert.assertEquals(2, res[1][1],0);
	}
	
	@Test
	public void testLarge()
	{
		long startInit = System.currentTimeMillis();
		
		double[][] A = Generator.create(60000, 785);
		double[][] B = Generator.create(785, 30);
		
		long endInit = System.currentTimeMillis();
		System.out.println("init:"+(endInit - startInit)+"ms");
		
		long startMult = System.currentTimeMillis();
		double [][]res = MatrixOperations.mult(A, B,2,Optional.empty());
		long endMult = System.currentTimeMillis();
		System.out.println("init:"+(endMult - startMult)+"ms");
		
		
		Assert.assertEquals(60000, res.length ,0);
		Assert.assertEquals(30, res[0].length,0);
	}
	
	@Test
	public void testVeryLarge()
	{
		long startInit = System.currentTimeMillis();
		
		double[][] A = Generator.create(150000, 785);
		double[][] B = Generator.create(785, 400);
		
		long endInit = System.currentTimeMillis();
		System.out.println("init:"+(endInit - startInit)+"ms");
		
		long startMult = System.currentTimeMillis();
		double [][]resThreaded = MatrixOperations.mult(A, B, 2,Optional.empty());
		long endMult = System.currentTimeMillis();
		System.out.println("very-large-2-threads:"+(endMult - startMult)/(double)60000+"m");
		
		Assert.assertEquals(150000, resThreaded.length ,0);
		Assert.assertEquals(400, resThreaded[0].length,0);
		
		//without multithreading
		
		startMult = System.currentTimeMillis();
		double[][]res = MatrixOperations.multSingleThread(A, B);
		endMult = System.currentTimeMillis();
		System.out.println("very-large-single-threads:"+(endMult - startMult)/(double)60000+"m");
		
		
		Assert.assertEquals(150000, res.length ,0);
		Assert.assertEquals(400, res[0].length,0);
		
		for(int i = 0; i < res.length;i++)
		{
			for(int j =0; j < res[0].length;j++)
			{
				Assert.assertEquals(resThreaded[i][j], res[i][j],0);
			}
		}
	}
	
	@Test
	public void testVeryLargeThreeThreads()
	{
		long startInit = System.currentTimeMillis();
		
		double[][] A = Generator.create(150000, 785);
		double[][] B = Generator.create(785, 400);
		
		long endInit = System.currentTimeMillis();
		System.out.println("init:"+(endInit - startInit)+"ms");
		
		long startMult = System.currentTimeMillis();
		double [][]resThreaded = MatrixOperations.mult(A, B, 3,Optional.empty());
		long endMult = System.currentTimeMillis();
		System.out.println("very-large-3-threads:"+(endMult - startMult)/(double)60000+"m");
		
		Assert.assertEquals(150000, resThreaded.length ,0);
		Assert.assertEquals(400, resThreaded[0].length,0);
		
		//without multithreading
		
		startMult = System.currentTimeMillis();
		double[][]res = MatrixOperations.multSingleThread(A, B);
		endMult = System.currentTimeMillis();
		System.out.println("very-large-single-threads:"+(endMult - startMult)/(double)60000+"m");
		
		
		Assert.assertEquals(150000, res.length ,0);
		Assert.assertEquals(400, res[0].length,0);
		
		for(int i = 0; i < res.length;i++)
		{
			for(int j =0; j < res[0].length;j++)
			{
				Assert.assertEquals(resThreaded[i][j], res[i][j],0);
			}
		}
	}
	
	@Test
	public void testLargeThreeThreads()
	{
		long startInit = System.currentTimeMillis();
		
		double[][] A = Generator.create(60000, 785);
		double[][] B = Generator.create(785, 30);
		
		long endInit = System.currentTimeMillis();
		System.out.println("init:"+(endInit - startInit)+"ms");
		
		long startMult = System.currentTimeMillis();
		double [][]resThreaded = MatrixOperations.mult(A, B, 3,Optional.empty());
		long endMult = System.currentTimeMillis();
		System.out.println("large-3-threads:"+(endMult - startMult)/(double)60000+"m");
		
		Assert.assertEquals(60000, resThreaded.length ,0);
		Assert.assertEquals(30, resThreaded[0].length,0);
		
		//without multithreading
		
		startMult = System.currentTimeMillis();
		double[][]res = MatrixOperations.multSingleThread(A, B);
		endMult = System.currentTimeMillis();
		System.out.println("large-single-threads:"+(endMult - startMult)/(double)60000+"m");
		
		
		Assert.assertEquals(60000, res.length ,0);
		Assert.assertEquals(30, res[0].length,0);
		
		for(int i = 0; i < res.length;i++)
		{
			for(int j =0; j < res[0].length;j++)
			{
				Assert.assertEquals(resThreaded[i][j], res[i][j],0);
			}
		}
	}
	
	@Test
	public void testSplitSmall()
	{
		long startInit = System.currentTimeMillis();
		
		double[][] A = Generator.createWithConstant(11, 4);
		
		long endInit = System.currentTimeMillis();
		System.out.println("init:"+(endInit - startInit)+"ms");
		
		long startMult = System.currentTimeMillis();
		double [][][]res = MatrixOperations.split(A, 5);
		long endMult = System.currentTimeMillis();
		System.out.println("split:"+(endMult - startMult)+"ms");
		
		
		Assert.assertEquals(2, res.length ,0);
		
		Assert.assertEquals(6, res[0].length,0);
		Assert.assertEquals(5, res[1].length,0);
		Assert.assertEquals(4, res[0][0].length,0);
		Assert.assertEquals(4, res[1][0].length,0);
		
		Assert.assertEquals(0, res[0][0][0],0);
		Assert.assertEquals(5, res[0][5][0],0);
		
		Assert.assertEquals(6, res[1][0][0],0);
		Assert.assertEquals(10, res[1][4][0],0);
	}
	
	@Test
	public void testSplitUneven()
	{
		long startInit = System.currentTimeMillis();
		
		double[][] A = Generator.createWithConstant(11, 4);
		
		long endInit = System.currentTimeMillis();
		System.out.println("init:"+(endInit - startInit)+"ms");
		
		long startMult = System.currentTimeMillis();
		double [][][]res = MatrixOperations.split(A, 11/2);
		long endMult = System.currentTimeMillis();
		System.out.println("split:"+(endMult - startMult)+"ms");
		
		
		Assert.assertEquals(2, res.length ,0);
		Assert.assertEquals(11, res[0].length + res[1].length,0);
		
		Assert.assertEquals(6, res[0].length,0);
		Assert.assertEquals(5, res[1].length,0);
		Assert.assertEquals(4, res[0][0].length,0);
		Assert.assertEquals(4, res[1][0].length,0);
		
		Assert.assertEquals(0, res[0][0][0],0);
		Assert.assertEquals(5, res[0][5][0],0);
		
		Assert.assertEquals(6, res[1][0][0],0);
		Assert.assertEquals(10, res[1][4][0],0);
	}
	
	@Test
	public void testSplitEven()
	{
		long startInit = System.currentTimeMillis();
		
		double[][] A = Generator.createWithConstant(12, 4);
		
		long endInit = System.currentTimeMillis();
		System.out.println("init:"+(endInit - startInit)+"ms");
		
		long startMult = System.currentTimeMillis();
		double [][][]res = MatrixOperations.split(A, 12/2);
		long endMult = System.currentTimeMillis();
		System.out.println("split:"+(endMult - startMult)+"ms");
		
		
		Assert.assertEquals(2, res.length ,0);
		Assert.assertEquals(12, res[0].length + res[1].length,0);
		
		Assert.assertEquals(7, res[0].length,0);
		Assert.assertEquals(5, res[1].length,0);
		Assert.assertEquals(4, res[0][0].length,0);
		Assert.assertEquals(4, res[1][0].length,0);
		
		Assert.assertEquals(0, res[0][0][0],0);
		Assert.assertEquals(5, res[0][5][0],0);
		
		Assert.assertEquals(7, res[1][0][0],0);
		Assert.assertEquals(11, res[1][4][0],0);
	}
	
	
	@Test
	public void testSplitVerylarge()
	{
		long startInit = System.currentTimeMillis();
		
		double[][] A = Generator.createWithConstant(150000, 785);
		
		long endInit = System.currentTimeMillis();
		System.out.println("init:"+(endInit - startInit)+"ms");
		
		long startMult = System.currentTimeMillis();
		double [][][]res = MatrixOperations.split(A, A.length/2);
		long endMult = System.currentTimeMillis();
		System.out.println("split:"+(endMult - startMult)+"ms");
		
		
		Assert.assertEquals(2, res.length ,0);
		Assert.assertEquals(150000, res[0].length + res[1].length,0);
		Assert.assertEquals(149999, res[1][74998][0],0);
		
		Assert.assertEquals(75001, res[0].length,0);
		Assert.assertEquals(74999, res[1].length,0);
		
		Assert.assertEquals(785, res[0][0].length,0);
		Assert.assertEquals(785, res[1][0].length,0);
	}
	
	@Test
	public void testLargeMatrixXVector()
	{
		double [][] A = Generator.create(100000, 1000);
		double [] v = Generator.createRow(1000);
		
		long startMult = System.currentTimeMillis();
		double []res = MatrixOperations.multMxCV(A,v, 2,Optional.empty());
		long endMult = System.currentTimeMillis();
		System.out.println("MatrixVector:"+(endMult - startMult)/(double)1000+"s");
		
		Assert.assertEquals(100000, res.length ,0);

	}
	
	
	@Test
	public void testLargeVectorXMatrix()
	{
		double [][] A = Generator.create(100000, 1000);
		
		double [] v = Generator.createRow(100000);
		
		long startMult = System.currentTimeMillis();
		double [][]res = MatrixOperations.multRVxM(v, A, 2,Optional.empty());
		long endMult = System.currentTimeMillis();
		System.out.println("Vector X Matrix:"+(endMult - startMult)/(double)60000+"ms");
		
		Assert.assertEquals(1, res.length ,0);
		Assert.assertEquals(1000, res[0].length ,0);

	}
	
	@Test
	public void testSmallVectorXMatrix()
	{
		double [][] A = new double[2][3];
		A[0] = new double[]{3,2,1};
		A[1] = new double[]{1,0,2};
		
		double [] v = new double[2];
		v = new double[]{1,2};
		
		double [][]res = MatrixOperations.multRVxM(v, A, 2,Optional.empty());
		
		Assert.assertEquals(1, res.length ,0);
		Assert.assertEquals(3, res[0].length ,0);
		
		Assert.assertEquals(5, res[0][0],0);
		Assert.assertEquals(2, res[0][1],0);
		Assert.assertEquals(5, res[0][2],0);

	}
	
	@Test
	public void testLargeVectorColxVectorRow()
	{
		double [] c = Generator.createRow(100000);
		double [] r = Generator.createRow(2000);
		
		double [][]res = MatrixOperations.multCVxRV(c,r, 2,Optional.empty());
		
		Assert.assertEquals(100000, res.length ,0);
		Assert.assertEquals(2000, res[0].length ,0);
	}
	
	@Test
	public void testLargeVectorRowxVectorCol()
	{
		double [] c = Generator.createRow(100000);
		double [] r = Generator.createRow(100000);
		
		double [][]res = MatrixOperations.multRVxCV(c,r, 2);
		
		Assert.assertEquals(1, res.length ,0);
		Assert.assertEquals(1, res[0].length ,0);
	}
	
	@Test
	public void testInsertConstVeryLarge()
	{
		double [][] c = Generator.create(700000, 300);

		long startMult = System.currentTimeMillis();
		double [][]res = MatrixOperations.insertConstFirstCol(c,1.0);
		long endMult = System.currentTimeMillis();
		System.out.println("Insert first col:"+(endMult - startMult)/(double)1000+"s");
		
		for(int i = 0; i < 700000;i++)
		{
			Assert.assertEquals(1.0, res[i][0],0);
		}
	}
	
	@Test
	public void testLargeTranspose()
	{
		double[][] A = Generator.create(60000, 785);

		long startMult = System.currentTimeMillis();
		double [][]resThreaded = MatrixOperations.transpose(A);
		long endMult = System.currentTimeMillis();
		System.out.println("transpose:"+(endMult - startMult)/(double)60000+"m");
		
		Assert.assertEquals(785, resThreaded.length ,0);
		Assert.assertEquals(60000, resThreaded[0].length,0);
		
		for(int i = 0; i < resThreaded.length;i++)
		{
			for(int j =0; j < resThreaded[0].length;j++)
			{
				Assert.assertEquals(resThreaded[i][j], A[j][i],0);
			}
		}
	}
		
	
}
