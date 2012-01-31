import java.io.IOException;

public class main{
	public static void main(String args[]) throws IOException{
		
		/*
		 * repeatedly run the following (), in different settings (1,2,3)
		 * 
		 * 1,generate training data
		 * 2,train
		 * 3,generate testing data
		 * 4,test
		 */
//		Environment c = new Environment();
		Vector4 v4 = new Vector4();
		iSVM lk = new iSVM();
		
//		for (int i = 0; i < Environment.dataSetNum; i++){
		for (int i = 0; i < 1; i++){
			v4.newData();
			lk.go(v4,Environment.dataSetSize,Environment.trainSize);
			lk.evaluation();
			
		}
		
//		for (int i = 5000; i < 5030; i ++){
//			v4.printV(i);
//		}
		
		
//		System.out.println(p);
		
//		double sum = 0;
//		for (int i = 0; i < 20000000; i ++)
//		{
//			sum += c.sampleNormalUnivariate(2,1);
//		}
//		System.out.println("sum = "+ sum +" ave = " + sum * 1.0 / 20000000);
		return;
	}
}