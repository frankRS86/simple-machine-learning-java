package la;

import org.junit.Assert;
import org.junit.Test;


public class SequenceTest {

	@Test
	public void testExecuteSinglePow()
	{
		double[][] A = new double[2][2];
		A[0] = new double[]{2,4};
		A[1] = new double[]{3,6};
		
		double[][]C = MatrixOperations.executeOnEachElement(A, MatrixOperations.POW2);
		
		Assert.assertEquals(2, C.length,0);
		Assert.assertEquals(2, C[0].length,0);
		
		Assert.assertEquals(4, C[0][0],0);
		Assert.assertEquals(16, C[0][1],0);
		Assert.assertEquals(9, C[1][0],0);
		Assert.assertEquals(36, C[1][1],0);
	}
	
	@Test
	public void testExecuteSingleSqrt()
	{
		double[][] A = new double[2][2];
		A[0] = new double[]{25,4};
		A[1] = new double[]{9,36};
		
		double[][]C = MatrixOperations.executeOnEachElement(A, MatrixOperations.SQRT);
		
		Assert.assertEquals(2, C.length,0);
		Assert.assertEquals(2, C[0].length,0);
		
		Assert.assertEquals(5, C[0][0],0);
		Assert.assertEquals(2, C[0][1],0);
		Assert.assertEquals(3, C[1][0],0);
		Assert.assertEquals(6, C[1][1],0);
	}
	
	@Test
	public void testExecute2PowSqrt()
	{
		double[][] A = new double[2][2];
		A[0] = new double[]{2,4};
		A[1] = new double[]{3,6};
		
		double[][]C = MatrixOperations.executeOnEachElement(A, MatrixOperations.POW2,MatrixOperations.SQRT);
		
		Assert.assertEquals(2, C.length,0);
		Assert.assertEquals(2, C[0].length,0);
		
		Assert.assertEquals(2, C[0][0],0);
		Assert.assertEquals(4, C[0][1],0);
		Assert.assertEquals(3, C[1][0],0);
		Assert.assertEquals(6, C[1][1],0);
	}
	
	@Test
	public void testExecute2PowPow()
	{
		double[][] A = new double[2][2];
		A[0] = new double[]{2,4};
		A[1] = new double[]{3,6};
		
		double[][]C = MatrixOperations.executeOnEachElement(A, MatrixOperations.POW2,MatrixOperations.POW2);
		
		Assert.assertEquals(2, C.length,0);
		Assert.assertEquals(2, C[0].length,0);
		
		Assert.assertEquals(16, C[0][0],0);
		Assert.assertEquals(256, C[0][1],0);
		Assert.assertEquals(81, C[1][0],0);
		Assert.assertEquals(1296, C[1][1],0);
	}
}
