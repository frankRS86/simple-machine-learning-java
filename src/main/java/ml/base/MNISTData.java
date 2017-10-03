package ml.base;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;


public class MNISTData {

	private int numLabels = 0;
	private int numImages = 0;
	private int numRows = 0;
	private int numCols = 0;
	private FeatureSet data = null;

	public MNISTData(String labelFilename, String imageFilename,int loadSize) {
		try {
			DataInputStream labels = new DataInputStream(new FileInputStream(
					labelFilename));
			DataInputStream images = new DataInputStream(new FileInputStream(
					imageFilename));
			int magicNumber = labels.readInt();
			if (magicNumber != 2049) {
				System.out.println("ERROR");
				return;
			}
			magicNumber = images.readInt();
			if (magicNumber != 2051) {
				System.out.println("ERROR");
				return;
			}
			this.numLabels = labels.readInt();
			this.numImages = images.readInt();
			this.numRows = images.readInt();
			this.numCols = images.readInt();
			if (numLabels != numImages) {
				StringBuilder str = new StringBuilder();
				str.append("Image file and label file do not contain the same number of entries.\n");
				str.append("  Label file contains: " + numLabels + "\n");
				str.append("  Image file contains: " + numImages + "\n");
				System.out.println("ERROR");
				return;
			}

			byte[] labelsData = new byte[numLabels];
			labels.read(labelsData);
			int imageVectorSize = numCols * numRows;
			byte[] imagesData = new byte[numLabels * imageVectorSize];
			images.read(imagesData);
			
			this.data = new FeatureSet();
			int imageIndex = 0;
			System.out.println("reading images. image size = "+imageVectorSize);
			
			int max = loadSize;
			System.out.println("loading images: "+max);
			double[][] d = new double [max][imageVectorSize];
			double[][] l = new double [max][10];
			
			for(int i=0;i<max;i++) {
				
				if(i % 1000 == 0)
				{
					//System.out.println("reading image "+i);
				}
				
				int label = labelsData[i];
				
				//List<Double> inputData = new LinkedList<Double>();
				double[] inputData = new double[imageVectorSize];
				for(int j=0;j<imageVectorSize;j++) {
					inputData[j] = (((double)(imagesData[imageIndex++]&0xff))/255.0);
				}
				double[] idealData = new double[10];
				for(int j = 0; j < 10;j++)
				{
					idealData[j] = (j==label?1.0:0.0);
				}
				d[i] = inputData;
				l[i] = idealData;
				//this.data.addExample(new FeatureVector("", Vector.fromCollection(idealData), inputData));
			}
			data.setFeatureMatrix(d);
			data.setLabelMatrix(l);
			System.out.println("finished parsing. closing streams");
			
			images.close();
			labels.close();
			

		} catch (IOException ex) {
			System.out.println("ERROR");
			return;
		}
	}

	/**
	 * @return the numLabels
	 */
	public int getNumLabels() {
		return numLabels;
	}

	/**
	 * @return the numImages
	 */
	public int getNumImages() {
		return numImages;
	}

	/**
	 * @return the numRows
	 */
	public int getNumRows() {
		return numRows;
	}

	/**
	 * @return the numCols
	 */
	public int getNumCols() {
		return numCols;
	}

	/**
	 * @return the data
	 */
	public FeatureSet getData() {
		return data;
	}
	
	

}
