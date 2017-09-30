package la;

import java.util.Optional;

import org.junit.Assert;
import org.junit.Test;

public class SliceTest 
{
	@Test
	public void testSliceOneCol()
	{
		double[][] A = new double[2][1];
		A[0] = new double[]{2};
		A[1] = new double[]{3};
		
		double[][] C = MatrixOperations.slice(A, 0,Optional.empty());
		
		Assert.assertEquals(0, C[0].length);
	}
	
	@Test
	public void testSlicetwoCol()
	{
		double[][] A = new double[2][2];
		A[0] = new double[]{2,4};
		A[1] = new double[]{3,6};
		
		double[][] C = MatrixOperations.slice(A, 0,Optional.empty());
		
		Assert.assertEquals(2, C.length);
		Assert.assertEquals(1, C[0].length);
		Assert.assertEquals(4, C[0][0],0);
		Assert.assertEquals(6, C[1][0],0);
	}
	
	@Test
	public void testSliceMiddle()
	{
		double[][] A = new double[2][3];
		A[0] = new double[]{2,4,8};
		A[1] = new double[]{3,6,12};
		
		double[][] C = MatrixOperations.slice(A, 1,Optional.empty());
		
		Assert.assertEquals(2, C.length);
		Assert.assertEquals(2, C[0].length);
		Assert.assertEquals(2, C[0][0],0);
		Assert.assertEquals(8, C[0][1],0);
		Assert.assertEquals(3, C[1][0],0);
		Assert.assertEquals(12,C[1][1],0);
	}
	
	@Test
	public void testSliceEnd()
	{
		double[][] A = new double[2][3];
		A[0] = new double[]{2,4,8};
		A[1] = new double[]{3,6,12};
		
		double[][] C = MatrixOperations.slice(A, 2,Optional.empty());
		
		Assert.assertEquals(2, C.length);
		Assert.assertEquals(2, C[0].length);
		Assert.assertEquals(2, C[0][0],0);
		Assert.assertEquals(4, C[0][1],0);
		Assert.assertEquals(3, C[1][0],0);
		Assert.assertEquals(6, C[1][1],0);
	}
	
	@Test
	public void testSlicePreEnd()
	{
		double[][] A = new double[2][3];
		A[0] = new double[]{2,4,8,16};
		A[1] = new double[]{3,6,12,24};
		
		double[][] C = MatrixOperations.slice(A, 2,Optional.empty());
		
		Assert.assertEquals(2, C.length);
		Assert.assertEquals(3, C[0].length);
		Assert.assertEquals(2, C[0][0],0);
		Assert.assertEquals(4, C[0][1],0);
		Assert.assertEquals(3, C[1][0],0);
		Assert.assertEquals(6, C[1][1],0);
		Assert.assertEquals(16, C[0][2],0);
		Assert.assertEquals(24, C[1][2],0);
	}
	
	@Test
	public void testSliceMiddleAndPow()
	{
		double[][] A = new double[2][3];
		A[0] = new double[]{2,4,8};
		A[1] = new double[]{3,6,12};
		
		double[][] C = MatrixOperations.slice(A, 1,Optional.of(MatrixOperations.POW2));
		
		Assert.assertEquals(2, C.length);
		Assert.assertEquals(2, C[0].length);
		Assert.assertEquals(4, C[0][0],0);
		Assert.assertEquals(64, C[0][1],0);
		Assert.assertEquals(9, C[1][0],0);
		Assert.assertEquals(144,C[1][1],0);
	}
}
