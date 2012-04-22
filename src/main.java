import java.io.IOException;

public class main{
	public static void main(String args[]) throws IOException{
		
		/*
		 * run the following in dataSets(), in different settings (1,2,3)
		 * 
		 * 1,generate training data
		 * 2,train
		 * 3,generate testing data
		 * 4,test
		 */
//		Environment c = new Environment();
		Setting1 v4 = new Setting1();
		iSVM lk = new iSVM();
//		double u1 = 2;
//		double theta = - 1 * Math.sin(u1 * u1 * u1 + 1.2) -u1 * Math.cos(1 * u1 + 0.7) - 1 * u1 + 2;
//		double p = 1 / (1 + Math.exp(-theta));
//		System.out.println("prob = "+p);
		
		for (int i = 0; i < Environment.dataSetNum; i++){
//			v4.newData();
			v4.oldData("jiayou!");
			lk.go(v4,Environment.dataSetSize,Environment.trainSize);
			lk.evaluation();
//			v4.testSample(1000000,1,0.0001);
		}
		lk.report();
		v4.printF();
		System.out.println(lk.changeTimes);
		
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