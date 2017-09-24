package ml.base;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.ByteBuffer;

import org.la4j.Matrix;
import org.la4j.Vector;

public class Computations
{
	public static Vector insertInVectorFront(Vector orig, double val)
	{
		Vector withBias = Vector.constant(orig.length() + 1, val);
		for (int i = 0; i < orig.length(); i++)
		{
			withBias.set(i + 1, orig.get(i));
		}
		return withBias;
	}

	public static double sigmoid(double z)
	{
		double g = 1.0 / (1.0 + Math.exp(-z));
		return g;
	}

	public static Vector sigmoid(Vector z)
	{
		z.update((c, v) -> Math.exp(-v));
		Vector added = z.add(1.0);

		Vector ones = Vector.constant(z.length(), 1);
		ones.update((i, v) -> 1 / added.get(i));

		return ones;
	}

	public static Matrix sigmoid(Matrix z)
	{
		z.update((c, i, v) -> Math.exp(-v));
		Matrix added = z.add(1.0);

		Matrix ones = Matrix.constant(z.rows(), z.columns(), 1);
		ones.update((i, j, v) -> 1 / added.get(i, j));

		return ones;
	}

	public static double sigmoidGradient(double z)
	{
		double g = sigmoid(z);
		double sg = g * (1 - g);
		return sg;
	}

	public static Vector sigmoidGradient(Vector z)
	{
		Vector sg = sigmoid(z);
		sg.update((i, v) -> v * (1 - v));
		return sg;
	}

	public static Matrix sigmoidGradient(Matrix z)
	{
		Matrix sg = sigmoid(z);
		sg.update((i, j, v) -> v * (1 - v));
		return sg;
	}

	public static double round(double value, int places)
	{
		if (places < 0)
			throw new IllegalArgumentException();

		BigDecimal bd = new BigDecimal(value);
		bd = bd.setScale(places, RoundingMode.HALF_UP);
		return bd.doubleValue();
	}

	public static double[] toDoubleArray(byte[] byteArray)
	{
		int times = Double.SIZE / Byte.SIZE;
		double[] doubles = new double[byteArray.length / times];
		for (int i = 0; i < doubles.length; i++)
		{
			doubles[i] = ByteBuffer.wrap(byteArray, i * times, times).getDouble();
		}
		return doubles;
	}
}
