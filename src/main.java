import java.io.IOException;

public class main{
	public static void main(String args[]) throws IOException{
		Environment c = new Environment();
		/*
		 * repeatedly run the following (), in different settings (1,2,3)
		 * 
		 * 1,generate training data
		 * 2,train
		 * 3,generate testing data
		 * 4,test
		 * 
		 */
		double sum = 0;
		for (int i = 0; i < 20000000; i ++)
		{
			sum += c.sampleStandardNormalUnivariate();
		}
		System.out.println("sum = "+ sum +" ave = " + sum * 1.0 / 20000000);
		return;
	}
}