package ml.kmeans;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;

import la.MatrixOperations;
import la.VectorOperations;
import ml.base.FeatureSet;

public class KMeans
{
	public List<double[]> initCentroids(int K, FeatureSet feature)
	{
		List<double[]> result = new ArrayList<double[]>();
		Random rand = new Random();

		for (int i = 0; i < K; i++)
		{
			int idx = rand.nextInt(feature.getExampleSize());
			result.add(feature.getExample(idx));
		}

		return result;
	}

	public KMeansResult train(int K, int repeats, FeatureSet X)
	{
		List<double[]> my = initCentroids(K, X);
		ClosestResult r = null;

		for (int i = 0; i < repeats; i++)
		{
			r = findClosestCentroids(X, my);

			my = cumputeCentroidMeans(X, r.cy, K);
		}

		return new KMeansResult(r.cy, my, r.J);
	}

	private List<double[]> cumputeCentroidMeans(FeatureSet featureSet, List<Integer> cy, int k)
	{
		int count = featureSet.getExampleSize();
		List<double[]> centroids = new ArrayList<double[]>(k);
		List<Integer> centroidAssCount = new ArrayList<Integer>(k);

		for (int x = 0; x < k; x++)
		{
			centroids.add(VectorOperations.zeros(featureSet.getExample(0).length));
			centroidAssCount.add(0);
		}

		for (int m = 0; m < count; m++)
		{
			int centroidIdx = cy.get(m);
			double[] example = featureSet.getExample(m);

			centroidAssCount.set(centroidIdx, centroidAssCount.get(centroidIdx) + 1);
			centroids.set(centroidIdx, VectorOperations.add(centroids.get(centroidIdx),example));
		}

		for (int i = 0; i < k; i++)
		{
			centroids.set(i, VectorOperations.divide(centroids.get(i),centroidAssCount.get(i)));
		}

		return centroids;
	}

	private ClosestResult findClosestCentroids(FeatureSet X, List<double[]> my)
	{
		int count = X.getExampleSize();
		List<Integer> assignments = new ArrayList<Integer>(count);
		for (int a = 0; a < count; a++)
		{
			assignments.add(-1);
		}

		double J = 0;

		for (int i = 0; i < count; i++)
		{
			double[] example = X.getExample(i);
			double min = Double.MAX_VALUE;
			for (int y = 0; y < my.size(); y++)
			{
				double[] centroid = my.get(y);
				double[] diff = VectorOperations.subtract(example,centroid,MatrixOperations.POW2);
				double dist = VectorOperations.sum(diff);

				if (dist < min)
				{
					min = dist;
					assignments.set(i, y);
				}
			}

			J += min;
		}

		J /= count;

		return new ClosestResult(J, assignments);
	}

	static class ClosestResult
	{
		public double J;
		public List<Integer> cy;

		public ClosestResult(double j, List<Integer> cy)
		{
			super();
			J = j;
			this.cy = cy;
		}

	}

}
