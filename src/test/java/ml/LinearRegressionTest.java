package ml;

import org.junit.Test;

import ml.base.FeatureSet;
import ml.linearreg.LinearRegression;

public class LinearRegressionTest 
{
	@Test
	public void testPredictHouses()
	{
		double [][]X = new double[47][2];
		double []y = new double[47];
		
		X[0] = new double []{2104,3};
		X[1] = new double []{1600	,3};
		X[2] = new double []{2400	,3};
		X[3] = new double []{1416,	2};
		X[4] = new double []{3000,	4};
		X[5] = new double []{1985	,4};
		X[6] = new double []{1534,	3};
		X[7] = new double []{1427,	3};
		X[8] = new double []{1380,	3};
		X[9] = new double []{1494	,3};
		X[10] = new double []{1940,	4};
		X[11] = new double []{2000	,3};
		X[12] = new double []{1890,	3};
		X[13] = new double []{4478,	5};
		X[14] = new double []{1268,	3};
		X[15] = new double []{2300,	4};
		X[16] = new double []{1320,	2};
		X[17] = new double []{1236	,3};
		X[18] = new double []{2609,	4};
		X[19] = new double []{3031,	4};
		X[20] = new double []{1767,	3};
		X[21] = new double []{1888,	2};
		X[22] = new double []{1604,	3};
		X[23] = new double []{1962,	4};
		X[24] = new double []{3890,	3};
		X[25] = new double []{1100	,3};
		X[26] = new double []{1458,	3};
		X[27] = new double []{2526,	3};
		X[28] = new double []{2200,	3};
		X[29] = new double []{2637,	3};
		X[30] = new double []{1839,	2};
		X[31] = new double []{1000,	1};
		X[32] = new double []{2040,	4};
		X[33] = new double []{3137,	3};
		X[34] = new double []{1811,	4};
		X[35] = new double []{1437,	3};
		X[36] = new double []{1239,	3};
		X[37] = new double []{2132,	4};
		X[38] = new double []{4215,	4};
		X[39] = new double []{2162	,4};
		X[40] = new double []{1664,	2};
		X[41] = new double []{2238	,3};
		X[42] = new double []{2567	,4};
		X[43] = new double []{1200,	3};
		X[44] = new double []{852,	2};
		X[45] = new double []{1852,	4};
		X[46] = new double []{1203,	3};
	
		
		y[0] = 399900;
		y[1] = 329900;
		y[2] = 369000;
		y[3] = 232000;
		y[4] = 539900;
		y[5] = 299900;
		y[6] = 314900;
		y[7] = 198999;
		y[8] = 212000;
		y[9] = 242500;
		y[10] = 239999;
		y[11] = 347000;
		y[12] = 329999;
		y[13] = 699900;
		y[14] = 259900;
		y[15] = 449900;
		y[16] = 299900;
		y[17] = 199900;
		y[18] = 499998;
		y[19] = 599000;
		y[20] = 252900;
		y[21] = 255000;
		y[22] = 242900;
		y[23] = 259900;
		y[24] = 573900;
		y[25] = 249900;
		y[26] = 464500;
		y[27] = 469000;
		y[28] = 475000;
		y[29] = 299900;
		y[30] = 349900;
		y[31] = 169900;
		y[32] = 314900;
		y[33] = 579900;
		y[34] = 285900;
		y[35] = 249900;
		y[36] = 229900;
		y[37] = 345000;
		y[38] = 549000;
		y[39] = 287000;
		y[40] = 368500;
		y[41] = 329900;
		y[42] = 314000;
		y[43] = 299000;
		y[44] = 179900;
		y[45] = 299900;
		y[46] = 239500;
		
		FeatureSet s = new FeatureSet();
		s.setFeatureMatrix(X);
		s.normaliseStandardDeviation();
		
		LinearRegression lr = new LinearRegression();
		lr.train(1000, 0.01, X, y);
		
		double [] ex = new double[]{1650, 3};
		ex = s.normaliseStandradDeviation(ex);
		
		double p = lr.predict(ex);
		System.out.println("prediced "+p);
	}
	
}
