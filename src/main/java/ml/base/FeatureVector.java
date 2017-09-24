package ml.base;

import java.util.ArrayList;
import java.util.List;

import org.la4j.Vector;

public class FeatureVector
{
	// the identifier of this example
	private String name;

	// label is the outcome/result of this feature vector
	private Vector label;

	private Vector features;

	public FeatureVector(String name, Vector label)
	{
		this(name, label, new ArrayList<Double>());
	}

	public FeatureVector(String name, Vector label, List<Double> features)
	{
		this.name = name;
		this.label = label;
		this.features = Vector.fromCollection(features);
	}

	public FeatureVector(String name, Vector label, double... features)
	{
		this.name = name;
		this.label = label;
		this.features = Vector.fromArray(features);
	}

	public FeatureVector(String string)
	{
		this(string, Vector.constant(0, 0), new ArrayList<Double>());
	}

	public Vector getFeatures()
	{
		return features;
	}

	public void setFeatures(List<Double> features)
	{
		this.features = Vector.fromCollection(features);
	}

	public String getName()
	{
		return name;
	}

	public Vector getLabel()
	{
		return label;
	}

	public void addFeature(double f)
	{
		features = features.copyOfLength(features.length() + 1);
		features.set(features.length() - 1, f);
	}
}
