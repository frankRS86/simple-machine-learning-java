package ml.base;

import java.util.ArrayList;
import java.util.List;

import org.la4j.Matrix;
import org.la4j.Vector;

public class FeatureSet
{
	Matrix featureMatrix;

	List<String> exampleIds;

	List<Vector> ls;

	// contains mean value of each column
	Vector means;

	// Contains the max value of each column
	Vector max;

	private Matrix labelMatrix;

	public FeatureSet()
	{
		exampleIds = new ArrayList<String>();
	}

	/**
	 * adds a feature vector, representing a single training example
	 */
	public void addExample(FeatureVector example)
	{
		Vector data = example.getFeatures();

		if (exampleIds.size() > 0)
		{
			if (featureMatrix.columns() != data.length())
				throw new IllegalArgumentException("vector size of new example does not match old examples");
		}

		exampleIds.add(0, example.getName());
		Vector label = example.getLabel();
		if (labelMatrix == null)
		{
			labelMatrix = Matrix.zero(1, label.length());
			labelMatrix.setRow(0, label);
		}
		else
		{
			labelMatrix = labelMatrix.insertRow(0, label);
		}

		if (featureMatrix == null)
		{
			featureMatrix = Matrix.zero(1, data.length());
			featureMatrix.setRow(0, data);
			return;
		}

		featureMatrix = featureMatrix.insertRow(0, data);
	}

	/**
	 * this overrwrites all previously added training sets
	 */
	public void setAllData()
	{

	}

	public void normalise()
	{
		long start = System.currentTimeMillis();

		for (int i = 0; i < featureMatrix.columns(); i++)
		{
			Vector column = featureMatrix.getColumn(i);
			double max = column.max();
			double mean = column.sum() / column.length();

			Vector normalized = column.subtract(mean).divide(max);

			featureMatrix.setColumn(i, normalized);
		}

		long end = System.currentTimeMillis();
		System.out.println("normalization took ms:" + (end - start));
	}

	public Matrix getFeatureMatrix()
	{
		return featureMatrix;
	}

	public String getFeatureExampleName(int index)
	{
		if (index >= exampleIds.size())
		{
			return "INVALID";
		}
		return exampleIds.get(index);
	}

	public int getExampleSize()
	{
		return featureMatrix.rows();
	}

	public Vector getExample(int idx)
	{
		return featureMatrix.getRow(idx);
	}

	public void setLabelMatrix(Matrix labels)
	{
		this.labelMatrix = labels;
	}

	public Matrix getLabelMatrix()
	{
		return labelMatrix;
	}
}
