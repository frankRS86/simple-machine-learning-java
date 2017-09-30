package ml.kmeans;

import java.util.List;


public class KMeansResult
{
	// assignemnt if trainingData to centroid index
	// e.g. assignment(3) = 5 means training example three is assigned to
	// centroid 5
	public List<Integer> assignemnts;

	// the final centroid vectors
	public List<double[]> centroids;

	// the overall cost for this result
	public double J;

	public KMeansResult(List<Integer> assignemnts, List<double[]> centroids, double J)
	{
		super();
		this.assignemnts = assignemnts;
		this.centroids = centroids;
		this.J = J;
	}

}
