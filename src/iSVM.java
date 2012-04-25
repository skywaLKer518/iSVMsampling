import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.NumberFormat;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/*
 * model: infinite SVM
 * first do the discriminative model part.(generative part later)
 * 
 * Sample (/eta,z) together 
 * use stochastic approximation algorithm to get optimal parameters
 */
public class iSVM {
	private static final double stoppingCriterion = 1 ;
	private static final int maxIteration = 36000;
	private static final double deltaL = 1.0;
	private static final int paraSize = Environment.dataCateNum * Environment.trainSize;
	private static final int sampleNum = 200; // TODO
	private static final double alphaDP = 0.5;
	private static final double betaDP = 1.0;
	private static final double C = 3.0;
	private static final int modelNum = 200;
	private final int etaLength1 = 8; // for data setting 1
	private final int gammaLength1 = 4;
	private double logPartition = 0;
	private double eta[][];
	private double gamma[][];
	private int z[];	
	private double w[];					// w_d1_y1 w_d1_y2 ... w_d2_y1 w_d2_y2 ...
	private double EF[];
	private double G[];
//	private double f[];
	private double l[];
	private double mF[][];
	private double logV[];
	private double logP_0minusLogQ[];
	private double logVmax;
	private double u[];
	private Vector8 miu[];
	
	
	double pCompenent[] = new double[Environment.maxComponent]; // in test()
	
	
	private final int R = 15; // repeating times in Algorithm 5

	
	
	private int prediction[];
	private int score = 0;
	private List<Double> acc = new LinkedList<Double>();
	private double accuracy = 0;
	private double average = 0;
	//test
	private int data0,data1,predic0,predic1;
	public int changeTimes = 0;
	private int change = 0;
	
	
	
	/*
	 * here use similar algorithm as in MCMC on DP Model, not Stick Breaking Process.
	 */
	private int cateNumber = 0; // number of category
	private int cateIndexMax = 0;
	private int number[]; // record n_-i,c.  number[MaxCateNumber]
	private int Observed = 0;
	private int minN = 1;
	private List<Integer> cateAlive = new LinkedList<Integer>(); // store all cate which is associated with at least one data sample
	
	
	iSVM(){
		w = new double[paraSize];
		EF = new double[paraSize];
		G = new double[paraSize];
		l = new double[paraSize];
		u = new double[sampleNum];
		logV = new double[sampleNum];
		logP_0minusLogQ = new double[sampleNum];
		miu = new Vector8[Environment.maxComponent];
		for (int i = 0; i < Environment.maxComponent; i++){
			miu[i] = new Vector8();
		}
		z = new int[Environment.dataSetSize];
		mF = new double[paraSize][];
		for (int i = 0;i  < paraSize; i++){
			mF[i] = new double[sampleNum];
		}
		eta = new double[sampleNum][];
		for (int i = 0; i < sampleNum; i++){
			eta[i] = new double[etaLength1];
		}
		gamma = new double[Environment.maxComponent][];
		for (int i = 0; i < Environment.maxComponent; i++){
			gamma[i] = new double[gammaLength1];
		}
//		number = new int[Environment.maxComponent+1];
//		for (int i = 0; i <= Environment.maxComponent; i++){
//			number[i] = 0;
//		}
		prediction = new int[Environment.dataSetSize];
		logVmax = 0;
	}

	public void go(Data v4, int setSize, int trainSize) {
		init(v4);
//		train(v4,setSize,trainSize);
//		trainAlg1_2(v4,setSize,trainSize);
//		trainAlg2_1(v4,setSize,trainSize);
//		trainAlg2_2(v4,setSize,trainSize);
//		trainAlg3_1(v4,setSize,trainSize);
		tmpuse();
		
//		for (int k = 0; k < 200; k++)
			test(v4,setSize,trainSize);
//			test2(v4,setSize,trainSize);
//		System.out.println("score is "+score+" out of "+200 * 100+" accu = "+ score * 1.0 / (1.0 * 200 * 100));
//		
	}

	
	void tmpuse(){
//		Log outLog = new Log("deltaAAA.txt");
//		BufferedReader br = null;
//		FileInputStream fis = null;
//		InputStreamReader isr = null;
//		
//		try {
//			fis = new FileInputStream("deltaANDlogZ.txt");
//		} catch (FileNotFoundException e) {
//			e.printStackTrace();
//		}
//		isr = new InputStreamReader(fis);
//		br = new BufferedReader(isr);
//		String line = new String();
//		String [] s = new String[4];
//		try {
//			
//			for (int i = 0; i < 50; i++){
//				line = br.readLine();
//				line = br.readLine();
//				
//				s = line.split(" ");
//				outLog.outln(s[3]);
//				line = br.readLine();
//			}
//			
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
//		
//		outLog.close();
	}
	/*
	 * train parameter w[], (/eta and Z)
	 * stochastic approximation w = w - a_k * G(w)
	 * set a_k = 1/k
	 * G(w) = B - EF
	 * B_y,d = l_y,d,
	 * 
	 * logV = FW + logp_0 - logq_0 = FW
	 * EF = 1/(nZ) * FV
	 * Z = 1/n * sigma V
	 */
	private void train(Data v4, int setSize, int trainSize) {
		int k = 1; // k_th iteration
		//test
		Log log2 = new Log("EF.txt");
		double logVt[] = new double[sampleNum];
		Log lv = new Log("logVfile.txt");
		do {
			// sample N (z,/eta) compute matrix mF
			for (int j = 0; j < sampleNum; j++){
				logP_0minusLogQ[j] = 0;
				/*
				 * importance sample 1
				 * proposal distribution P_0(z) * p_0(/eta)
				 */
				//samplePara();
				/*
				 * importance sample 2
				 * proposal distribution changed
				 */
				samplePara2(v4,w,j);
				computeF(v4,j);
			}

			
			// computer logV (k*1 vector, storing importance weight)
			logVmax = 0;
			for (int j = 0; j < sampleNum; j++){
				logV[j] = 0;
				for (int i = 0; i < paraSize; i++){
					// logV = mF` * W
					logV[j] += mF[i][j] * w[i];
				}
				//test
//				System.out.println();
//				System.out.println("before, LogV[j] = "+logV[j]);
//				logV[j] += logP_0minusLogQ[j];
//				System.out.println("logP_0 - LogQ = "+logP_0minusLogQ[j]);
//				System.out.println("after, LogV[j] = "+logV[j]);
//				System.out.println();
				// record logVmax, (avoiding numerical overflow)
				if (logV[j] > logVmax)
					logVmax = logV[j];
				lv.outln("logV["+j+"]= " +logV[j]);
				logVt[j] = logV[j];
				System.out.println(logV[j]);
			}
			
			// test all the way
			double t;
			for (int i = 0; i < sampleNum; i++){
				for (int j = 0; j < sampleNum - i - 1; j++){
					if (logVt[j] < logVt[j + 1]){
						t = logVt[j];
						logVt[j] = logVt[j+1];
						logVt[j+1] = t;
					}
				}
			}
			lv.out3ln();
			lv.outln("after sorting");
			for (int i = 0; i < sampleNum; i ++){
				lv.outln(i);
				lv.outln("       "+logVt[i]);
			}
			lv.out3ln();
			// computer log partition function
			logPartition = logVmax;
			double sum = 0;
			for (int j = 0; j < sampleNum; j++){
				sum += Math.exp(logV[j] - logVmax);
			}
			logPartition += Math.log(sum);
			logPartition -= Math.log(sampleNum*1.0);
			
			// computer EF
			//test
			double sumU = 0,largeU = 0; // test 
			int numU = 0,index = 0; // test, too
			for(int j = 0; j < sampleNum; j ++){
				u[j] = Math.exp(logV[j] - logPartition);
				sumU += u[j];
				if (u[j] > largeU) {
					largeU = u[j];
					index = j;
				}
			}		
			//test
			for (int j = 0; j < sampleNum;j++){
				if (u[j] / largeU > 0.05) numU ++;
			}
//			for (int j = 0; j < sampleNum;j++){
//				System.out.println("u["+j+"] = " + u[j]);
//			}
			System.out.println("sumU = "+sumU +" largeU = " +largeU+" at "+index +" numU = "+numU);
			if (k > 10) break;  // test TODO
			
			for (int i = 0; i < paraSize; i ++){
				EF[i] = 0;
				for (int j = 0; j < sampleNum; j ++){
					EF[i] += mF[i][j] * u[j];
				}
				EF[i] = EF[i] / (1.0 * sampleNum);
			}
			// (G~) = (U~) - B
			double delta = 0;
			for (int i = 0; i < paraSize; i++){
				G[i] = EF[i] - l[i];
				delta += G[i] * G[i];
			}
				System.out.println("k = "+k+"; ");
				System.out.println("delta  = " + delta);
//			if ( k % 20 == 0){
				log2.outln("k = "+k);
				for (int i = 0; i < paraSize; i++){
					log2.outln("EF["+i+"] = " + EF[i]+ " l["+i+"] = "+l[i]);
				}
				log2.outln("u[1-sampleNum] = ");
				for (int q = 0; q < sampleNum; q++){
					log2.out(u[q]+" ");
				}
				log2.outln("");
				log2.outln("");log2.outln("");log2.outln("");
//			}
				
			if (delta < stoppingCriterion)
				break;
			double m = (1 / Math.pow((1.0 * k),2.0 / 3.0));
			for (int i = 0; i < paraSize; i++){
				w[i] -= m * G[i];
				if (w[i] > C)					w[i] = C;
				if (w[i] < 0)					w[i] = 0;
			}
			k++;
		}
		while (true);
		lv.close();
		log2.close();
		trainEndLog();
	}



	/*
	 * Stochastic Approximation to optimize w
	 * still importance sampling, only sample Z, from DP prior of Z
	 */
	private void trainAlg1_2(Data v4, int setSize, int trainSize) {
		int k = 1; // k_th iteration
		// test
		double logVt[] = new double[sampleNum];
		Log lv = new Log("logVfileAlg2_1.txt");
		Log log2 = new Log("EFAlg1_2.txt");
		do
		{
			// a, sample Z
			for (int j = 0; j < sampleNum; j++){
				int zMin = 99999, zMax = -1;
				boolean used[] = new boolean[Environment.maxComponent];
				double tmp[] = new double[8];
				Vector8 r = new Vector8();
				
				// 1, sample Z
				for (int i = 0; i < Environment.trainSize; i++){
					z[i] = samplePriorZ();
					if (z[i] > zMax) zMax = z[i];
					if (z[i] < zMin) zMin = z[i];
					used[z[i]] = true;
				}
				// 2, compute miu
				int yy = 0;
//				int [] num = new int[Environment.maxComponent];
				for (int i = zMin; i <= zMax; i++){
					if (!used[i]) continue;
					miu[i].init();
					for (int d = 0; d < Environment.trainSize; d++){
						if (z[d] != i) continue;
//						num[i]++;
						int yd = ((Setting1) v4).getLabel(d);
						if (yd == 0) yy = 1;
						else yy = 0;
						tmp = ((Setting1) v4).deltaF_d(d);
						r.setValue(tmp);
						r.multiply(w[2 * d + yy]);//test 
						miu[i].add(r);
					}
				}
				// 3, compute logV[j]
				logV[j] = 0;
				for (int i = zMin; i <= zMax; i++){
					if (!used[i]) continue;
					logV[j] += 0.5 * miu[i].multiply(miu[i]);
				}
				logVt[j] = logV[j]; // test
				// 4, compute mF
				for( int i = 0; i < paraSize; i++){
					int d = i / Environment.dataCateNum;
					int y = i % Environment.dataCateNum;
					int yd = ((Setting1) v4).getLabel(d);
					if (yd == y) mF[i][j] = 0;
					else{
						tmp = ((Setting1) v4).deltaF_d(d);
						mF[i][j] = miu[z[d]].multiply(tmp);
						// alternative implement way
//						double []tmp2 = new double [etaLength1];
//						tmp2 = miu[z[d]].getValue();
//						mF[i][j] = ((Setting1) v4).computeF(tmp2,d,y,yd);
					}
				}
			}
			// test all the way
			double t;
			for (int i = 0; i < sampleNum; i++){
				for (int j = 0; j < sampleNum - i - 1; j++){
					if (logVt[j] < logVt[j + 1]){
						t = logVt[j];
						logVt[j] = logVt[j+1];
						logVt[j+1] = t;
					}
				}
			}
			lv.out3ln();
			lv.outln("after sorting");
			for (int i = 0; i < sampleNum; i ++){
				lv.outln(i);
				lv.outln("       "+logVt[i]);
			}
			lv.out3ln();
			
			// b, compute partition function
			logVmax = 0;
			for (int j = 0; j < sampleNum; j++){
				if (logV[j] > logVmax)
					logVmax = logV[j];
			}
			logPartition = logVmax;
			double sumT = 0;
			for (int j = 0; j < sampleNum; j++)
				sumT += Math.exp(logV[j] - logVmax);
			logPartition += Math.log(sumT);
			logPartition -= Math.log(sampleNum*1.0); 
			
			// c, compute EF
			for(int j = 0; j < sampleNum; j ++){
				u[j] = Math.exp(logV[j] - logPartition);
			}
			for (int i = 0; i < paraSize; i++){
				EF[i] = 0;
				for (int j = 0; j < sampleNum; j++){
					EF[i] += mF[i][j] * u[j];
				}
				EF[i] = EF[i] / (1.0 * sampleNum);
			}
			
			// d, update w
			double delta = 0;
			for (int i = 0; i < paraSize; i++){
				G[i] = EF[i] - l[i];
				delta += G[i] * G[i];
			}
			// test all the way
			System.out.println("k = "+k+"; ");
			System.out.println("delta  = " + delta);
	
			log2.outln("k = "+k);
			for (int i = 0; i < paraSize; i++){
				log2.outln("EF["+i+"] = " + EF[i]+ " l["+i+"] = "+l[i]);
			}
			log2.outln("u[1-sampleNum] = ");
			for (int q = 0; q < sampleNum; q++){
				log2.out(u[q]+" ");
			}
			log2.out3ln();
			//
			if (delta < stoppingCriterion)
				break;
			double m = (1 / Math.pow((1.0 * k),2.0 / 3.0));
			for (int i = 0; i < paraSize; i++){
				w[i] -= m * G[i];
				if (w[i] > C)					w[i] = C;
				if (w[i] < 0)					w[i] = 0;
			}
			k++;
			
			if (k > 10) break;  // test TODO
			
		}while (true);
		lv.close();
		log2.close();
		trainEndLog();
		
		return;
	}
	
	/*
	 * Stochastic Approximation to optimize w
	 * Metropolis Sampling, sample Z
	 */
	private void trainAlg2_1(Data v4, int setSize, int trainSize) {
		
//		readW(new String("res/paraW(delta300.0k=2000).txt"));
		
		int k = 1; // k_th iteration
		// test
		double logVt[] = new double[sampleNum];
//		
		Log lv = new Log("res/delta(delta"+stoppingCriterion+"k"+maxIteration+").txt");
		Log log2 = new Log("res/EFAlg2_1(delta"+stoppingCriterion+"k"+maxIteration+").txt");
		Log log3 = new Log("res/sampleFAlg2_1(delta"+stoppingCriterion+"k"+maxIteration+").txt");
	
		// ~test

		// compute f_delta for all d,y
		Vector8[] f_delta = new Vector8[paraSize];
		double [] tmpp = new double[8];
		for (int i = 0; i < paraSize; i++){
			f_delta[i] = new Vector8();
			int dd = i / 2;
			int yy = i % 2;
			int y = ((Setting1) v4).getLabel(dd);
			if ( y == yy ) {}// do nothing
			else{
				tmpp = ((Setting1) v4).deltaF_d(dd);
				f_delta[i].setValue(tmpp);
			}
		}
		
		
		do
		{
//			if (k <= 2){
//				for (int i = 0; i < 20; i++){
//					((Setting1) v4).printV(i);
//					System.out.println("w[2*"+i+"] = "+w[2*i]);
//					System.out.println("w[2*"+i+"+1] = "+w[2*i+1]);
//				}
//			}
			if (k > maxIteration)
				break;
			
			// a, sample Z, using M-H Alg.
			//    compute mF[][]
			MCMC a = new MCMC(v4, trainSize, w, alphaDP);
			a.go();
			
			log3.outln("k = "+k+"; mF[0][j] ---");
			for (int i = 0; i < sampleNum; i++){
				a.oneSample();
				z = a.getZ();
				cateIndexMax = a.getCateNumer();
				for (int l = 1; l <= cateIndexMax; l ++){
//					miu[l].reset();
					miu[l] = a.getMiu(l);
				}
				for (int j = 0; j < paraSize; j++){
					int dd = j/2;
					mF[j][i] = miu[z[dd]].multiply(f_delta[j]);
				}
				log3.outln(mF[0][i]);
			}
			
			// b, compute EF
			for (int j = 0; j < paraSize; j++){
				EF[j] = 0;
				for (int i = 0; i < sampleNum; i++){
					EF[j] += mF[j][i];
				}
				EF[j] /= 1.0 * sampleNum;
			}
			// c, update w
			double delta = 0;
			for (int i = 0; i < paraSize; i++){
				G[i] = EF[i] - l[i];
				delta += G[i] * G[i];
			}
			// test all the way
			System.out.println("k = "+k+"; ");
			System.out.println("delta  = " + delta);
			if ( k % 100 == 1)
				lv.outln("k = "+k);
			NumberFormat num = NumberFormat.getInstance();
			num.setMaximumFractionDigits(0);
			lv.outln("delta  = " + num.format(delta));
			if ( k % 100 == 0)
				lv.outln("\n\n");
			
			log2.outln("k = "+k);
			for (int i = 0; i < paraSize; i++){
				log2.outln("EF["+i+"] = " + EF[i]+ " l["+i+"] = "+l[i]);
			}
//			log2.outln("u[1-sampleNum] = ");
//			for (int q = 0; q < sampleNum; q++){
//				log2.out(u[q]+" ");
//			}
			log2.out3ln();
			//
			if (delta < stoppingCriterion)
				break;
			double m = (1 / 500.0 / Math.pow((1.0 * k),2.0 / 3.0));
			for (int i = 0; i < paraSize; i++){
				w[i] -= m * G[i];
				if (w[i] > C)					w[i] = C;
				if (w[i] < 0)					w[i] = 0;
			}
//			// see w
//			if (k == 1){
//				Log changeW = new Log("updatedW.txt");
//				for (int i = 0; i < trainSize; i++){
//					int b = ((Setting1) v4).getLabel(i);
//					if (b == 0) b=1;
//					else b = 0;
//					changeW.outln("w[2*"+i+"+"+b+"] = " + w[2*i+b]);
//				}
//				changeW.close();
//			}
			k++;
			
			
		}while (true);
		lv.close();
		log2.close();
		log3.close();
		trainEndLog();
		
		return;
	}
	
	
	/*
	 * Stochastic Approximation to optimize w
	 * Gibbs Sampler, sample Z
	 */
	private void trainAlg2_2(Data v4, int setSize, int trainSize) {
//		readW(new String("res/paraW(delta10.0k=5000Times200).txt"));
		int k = 1; // k_th iteration
		// test
		double logVt[] = new double[sampleNum];
		Log lv = new Log("res/deltaAlg2_2(delta"+stoppingCriterion+"k"+maxIteration+"Times"+Environment.sampleTimes+").txt");
		Log logz = new Log("res/logZAlg2_2(delta"+stoppingCriterion+"k"+maxIteration+"Times"+Environment.sampleTimes+").txt");
		Log log2 = new Log("res/EFAlg2_2(delta"+stoppingCriterion+"k"+maxIteration+"Times"+Environment.sampleTimes+").txt");
		Log log3 = new Log("res/sampleFAlg2_2(delta"+stoppingCriterion+"k"+maxIteration+"Times"+Environment.sampleTimes+").txt");
		Log obj = new Log("res/objAlg2_2(delta"+stoppingCriterion+"k"+maxIteration+"Times"+Environment.sampleTimes+").txt");
		// ~test

		// compute f_delta for all d,y
		Vector8[] f_delta = new Vector8[paraSize];
		double [] tmpp = new double[8];
		for (int i = 0; i < paraSize; i++){
			f_delta[i] = new Vector8();
			int dd = i / 2;
			int yy = i % 2;
			int y = ((Setting1) v4).getLabel(dd);
			if ( y == yy ) {}// do nothing
			else{
				tmpp = ((Setting1) v4).deltaF_d(dd);
				f_delta[i].setValue(tmpp);
			}
		}
		do
		{
			// a, sample Z, using M-H Alg.
			//    compute mF[][]
			MCMC a = new MCMC(v4, trainSize, w, alphaDP);
			a.go2();
			
			log3.outln("k = "+k+"; mF[0][j] ---");
			for (int i = 0; i < sampleNum; i++){
				a.oneSample2();
				z = a.getZ();
				cateIndexMax = a.getCateNumer();
				for (int l = 1; l <= cateIndexMax; l ++){
//					miu[l].reset();
					miu[l] = a.getMiu(l);
				}
				for (int j = 0; j < paraSize; j++){
					int dd = j/2;
					mF[j][i] = miu[z[dd]].multiply(f_delta[j]);
				}
				log3.outln(mF[0][i]);
			}
			
			// b, compute EF
			for (int j = 0; j < paraSize; j++){
				EF[j] = 0;
				for (int i = 0; i < sampleNum; i++){
					EF[j] += mF[j][i];
				}
				EF[j] /= 1.0 * sampleNum;
			}
			// c, update w
			double delta = 0;
			for (int i = 0; i < paraSize; i++){
				G[i] = EF[i] - l[i];
				delta += G[i] * G[i];
			}
			// test all the way
			System.out.println("k = "+k+"; ");
			System.out.println("delta  = " + delta);
			if ( k % 100 == 1)
				lv.outln("k = "+k);
			NumberFormat num = NumberFormat.getInstance();
			num.setMaximumFractionDigits(0);
//			lv.outln(num.format(delta));
			lv.outln(delta);
			if ( k % 100 == 0)
				lv.outln("\n\n");
			
			log2.outln("k = "+k);
			for (int i = 0; i < paraSize; i++){
				log2.outln("EF["+i+"] = " + EF[i]+ " l["+i+"] = "+l[i]);
			}
			log2.outln("u[1-sampleNum] = ");
			for (int q = 0; q < sampleNum; q++){
				log2.out(u[q]+" ");
			}
			log2.out3ln();
			//double z = ApproximateNormal(w,);
//			int samNum = 100;
//			double logMax = 0, logZ, Z, sum = 0;
//			double[] logVtt = new double[samNum];
//			for (int i = 0; i < samNum; i++){
//				logVtt[i] = 0;
//				a.importanceS();
//				logVtt[i] = a.getSum();
////				System.out.println("logv["+i+"] ="+logVtt[i]);
//				if (logVtt[i] > logMax)
//					logMax = logVtt[i];
//			}
//			logZ = logMax - Math.log(samNum);
//			for (int i = 0; i < samNum; i++){
//				sum += Math.exp(logVtt[i] - logMax);
//			}
//			logZ += Math.log(sum);
//			System.out.println("logZ = "+logZ);
//			logz.outln(logZ);
//			
//			double WB = 0;
//			for (int i = 0; i < paraSize; i++){
//				WB += w[i] * l[i];
//			}
//			System.out.println("WB = \n" + WB);
//			double objective = WB - logZ;
//			System.out.println("objective " + objective);
//			obj.outln(objective);
			//
			if (delta < stoppingCriterion || k > maxIteration){
				break;
			}
			double m = (1 / 500.0 /  Math.pow((1.0 * k),2.0 / 3.0));
			for (int i = 0; i < paraSize; i++){
				w[i] -= m * G[i];
				if (w[i] > C)					w[i] = C;
				if (w[i] < 0)					w[i] = 0;
			}
//			// see w
//			if (k == 1){
//				Log changeW = new Log("updatedW.txt");
//				for (int i = 0; i < trainSize; i++){
//					int b = ((Setting1) v4).getLabel(i);
//					if (b == 0) b=1;
//					else b = 0;
//					changeW.outln("w[2*"+i+"+"+b+"] = " + w[2*i+b]);
//				}
//				changeW.close();
//			}
			k++;
		}while (true);
		lv.close();
		log2.close();
		log3.close();
		trainEndLog();
		logz.close();
		obj.close();
		return;
	}
	
	private void trainAlg3_1(Data v4, int setSize, int trainSize) {
		readW(new String("res/paraW(delta10.0k=1000Times200).txt"));
		int k = 1; // k_th iteration
		// test
		double logVt[] = new double[sampleNum];
		Log lv = new Log("res/deltaAlg3_1(delta"+stoppingCriterion+"k"+maxIteration+"Times"+Environment.sampleTimes+").txt");
		Log logz = new Log("res/logZAlg3_1(delta"+stoppingCriterion+"k"+maxIteration+"Times"+Environment.sampleTimes+").txt");
		Log log2 = new Log("res/EFAlg3_1(delta"+stoppingCriterion+"k"+maxIteration+"Times"+Environment.sampleTimes+").txt");
		Log log3 = new Log("res/sampleFAlg3_1(delta"+stoppingCriterion+"k"+maxIteration+"Times"+Environment.sampleTimes+").txt");
		Log obj = new Log("res/objAlg3_1(delta"+stoppingCriterion+"k"+maxIteration+"Times"+Environment.sampleTimes+").txt");
		// ~test

		// compute f_delta for all d,y
		Vector8[] f_delta = new Vector8[paraSize];
		double [] tmpp = new double[8];
		for (int i = 0; i < paraSize; i++){
			f_delta[i] = new Vector8();
			int dd = i / 2;
			int yy = i % 2;
			int y = ((Setting1) v4).getLabel(dd);
			if ( y == yy ) {}// do nothing
			else{
				tmpp = ((Setting1) v4).deltaF_d(dd);
				f_delta[i].setValue(tmpp);
			}
		}
		do
		{
			// a, sample Z, using M-H Alg.
			//    compute mF[][]
			MCMC a = new MCMC(v4, trainSize, w, alphaDP);
			a.go3();
			
			log3.outln("k = "+k+"; mF[0][j] ---");
			for (int i = 0; i < sampleNum; i++){
				a.oneSample3();
				z = a.getZ();
				cateIndexMax = a.getCateNumer();
				for (int l = 1; l <= cateIndexMax; l ++){
					miu[l] = a.getMiu(l);
				}
				for (int j = 0; j < paraSize; j++){
					int dd = j/2;
					mF[j][i] = miu[z[dd]].multiply(f_delta[j]);
				}
				log3.outln(mF[0][i]);
			}
			
			// b, compute EF
			for (int j = 0; j < paraSize; j++){
				EF[j] = 0;
				for (int i = 0; i < sampleNum; i++){
					EF[j] += mF[j][i];
				}
				EF[j] /= 1.0 * sampleNum;
			}
			// c, update w
			double delta = 0;
			for (int i = 0; i < paraSize; i++){
				G[i] = EF[i] - l[i];
				delta += G[i] * G[i];
			}
			// test all the way
			System.out.println("k = "+k+"; ");
			System.out.println("delta  = " + delta);
			if ( k % 100 == 1)
				lv.outln("k = "+k);
			NumberFormat num = NumberFormat.getInstance();
			num.setMaximumFractionDigits(0);
//			lv.outln(num.format(delta));
			lv.outln(delta);
			if ( k % 100 == 0)
				lv.outln("\n\n");
			
			log2.outln("k = "+k);
			for (int i = 0; i < paraSize; i++){
				log2.outln("EF["+i+"] = " + EF[i]+ " l["+i+"] = "+l[i]);
			}
//			log2.outln("u[1-sampleNum] = ");
//			for (int q = 0; q < sampleNum; q++){
//				log2.out(u[q]+" ");
//			}
			log2.out3ln();
			
			//double z = ApproximateNormal(w,);
//			int samNum = 100;
//			double logMax = 0, logZ, Z, sum = 0;
//			double[] logVtt = new double[samNum];
//			for (int i = 0; i < samNum; i++){
//				logVtt[i] = 0;
//				a.importanceS();
//				logVtt[i] = a.getSum();
////				System.out.println("logv["+i+"] ="+logVtt[i]);
//				if (logVtt[i] > logMax)
//					logMax = logVtt[i];
//			}
//			logZ = logMax - Math.log(samNum);
//			for (int i = 0; i < samNum; i++){
//				sum += Math.exp(logVtt[i] - logMax);
//			}
//			logZ += Math.log(sum);
//			System.out.println("logZ = "+logZ);
//			logz.outln(logZ);
//			
//			double WB = 0;
//			for (int i = 0; i < paraSize; i++){
//				WB += w[i] * l[i];
//			}
//			System.out.println("WB = \n" + WB);
//			double objective = WB - logZ;
//			System.out.println("objective " + objective);
//			obj.outln(objective);
			
			//
			if (delta < stoppingCriterion || k > maxIteration){
				break;
			}
				
			double m = (1 / 500.0 /  Math.pow((1.0 * k),2.0 / 3.0));
			for (int i = 0; i < paraSize; i++){
				w[i] -= m * G[i];
				if (w[i] > C)					w[i] = C;
				if (w[i] < 0)					w[i] = 0;
			}
//			// see w
//			if (k == 1){
//				Log changeW = new Log("updatedW.txt");
//				for (int i = 0; i < trainSize; i++){
//					int b = ((Setting1) v4).getLabel(i);
//					if (b == 0) b=1;
//					else b = 0;
//					changeW.outln("w[2*"+i+"+"+b+"] = " + w[2*i+b]);
//				}
//				changeW.close();
//			}
			k++;
			
			
		}while (true);
		lv.close();
		log2.close();
		log3.close();
		trainEndLog();
		logz.close();
		obj.close();
		
		return;
	}
	// read w[] from file
	private void readW(String filename) {
		BufferedReader br = null;
		FileInputStream fis = null;
		InputStreamReader isr = null;
		
		try {
			fis = new FileInputStream(filename);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		isr = new InputStreamReader(fis);
		br = new BufferedReader(isr);
		
		try {
			String line = new String();
			line = br.readLine();
			for (int i = 0; i < 200; i++){
				line = br.readLine();
				String[] s = new String[3];
				s=line.split(" ");
				w[i] = Double.parseDouble(s[2]);
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	// used in train
	private void computeF(Data v4,int sample) {
		for (int i = 0; i < paraSize; i++){
			int d = i / Environment.dataCateNum;
			int y = i % Environment.dataCateNum;
			int yd = ((Setting1) v4).getLabel(d);
			if (yd == y) mF[i][sample] = 0;
			else{
				mF[i][sample] = ((Setting1) v4).computeF(eta[z[d]],d,y,yd);
			}
		}
	}

	private void samplePara() {
		int zMin = 99999, zMax = -1;
		boolean used[] = new boolean[Environment.maxComponent];
		// sample Z
		for (int i = 0; i < Environment.trainSize; i++){
			z[i] = samplePriorZ();
			if (z[i] > zMax) zMax = z[i];
			if (z[i] < zMin) zMin = z[i];
			used[z[i]] = true;
		}
		// sample /eta
//		int time = 0; // record how many components there is
		for (int i = zMin; i <= zMax; i ++){
			if (!used[i]) continue;
//			time ++;
			drawEta(i);
		}
//		System.out.println("time is : "+time);
	}
	
	/*
	 * sample parameters Z and /eta
	 * importance sampling
	 * proposal distribution changed to: 1,p_0(Z) 2, for each i, /eta_i ~ N(u,I)
	 * where u is the new mean, given by all data with component z
	 * 
	 * for the jth sample
	 */
	private void samplePara2(Data v4, double[] w2, int sample){
		int zMin = 99999, zMax = -1;
		boolean used[] = new boolean[Environment.maxComponent];
		Vector8[] miuT = new Vector8[Environment.maxComponent];
		
		for (int i = 0; i < Environment.maxComponent; i++){
			miuT[i] = new Vector8();
		}
		Vector8 r = new Vector8();
		double tmp[] = new double[8];
		
		// sample Z
		for (int i = 0; i < Environment.trainSize; i++){
			z[i] = samplePriorZ();
			if (z[i] > zMax) zMax = z[i];
			if (z[i] < zMin) zMin = z[i];
			used[z[i]] = true;
		}
		// test clustering
		int[] num = new int[100];
		// sample /eta
		int y = 0;
		for (int i = zMin; i <= zMax; i ++){
			if (!used[i]) continue;
			for (int j = 0; j < Environment.trainSize; j++){
				if (z[j] != i) continue;
				num[i]++;
				int yd = ((Setting1) v4).getLabel(j);
				if (yd == 0) y = 1;
				else y = 0;
				tmp = ((Setting1) v4).deltaF_d(j);
				r.setValue(tmp);
				r.multiply(w[2 * j + y]);//test
				miuT[i].add(r);
			}
			drawEta2(i,miuT[i]);
			logP_0minusLogQ[sample] += 0.5 * miuT[i].multiply(miuT[i]) - miuT[i].multiply(eta[i]);
//			System.out.println("half deltaMean square ("+i+") at sample ("+sample+") ="+0.5*miuT[i].multiply(miuT[i]));
		}
//		System.out.println(logP_0minusLogQ[sample]);
//		System.out.println();
		if (sample== 50){
			Log oS = new Log("oneSample.txt");
			for (int i = zMin; i <= zMax; i++){
				if (used[i]){
					oS.out(i+"("+num[i]+")  ");	
				}
			}
			int pos = 0, neg = 0;
			for (int i = 0;i < 100; i++){
				if (((Setting1) v4).getLabel(i) == 0)
					pos ++;
				else neg++;
			}
			oS.outln("");oS.outln("pos = "+pos+"  neg = "+neg);
			
			for (int i = zMin; i <= zMax; i++){
				if (used[i]){
					oS.out(eta[i],8);
				}
			}
			oS.close();
		}
	}
	/*
	 * draw eta[index] from N(v,I)
	 */
	private void drawEta2(int index, Vector8 v) {
		double r[] = new double [etaLength1];
		for (int i = 0 ; i < etaLength1 ; i++){
			r[i] = sampleNormalUnivariate(v.d[i],1);
		}
		vectorAssign(eta[index],etaLength1,r);
		return;
	}

	private int samplePriorZ() {
		double a = Math.random();
		double b = 1 / ( 1 + alphaDP );
		int target = 1;
		do {
			if (a < b) return target;
			target ++;
			a -= b;
			b *= alphaDP / (1 + alphaDP);
		}while (true);
	}

	private void init(Data v4) {
		for (int i = 0; i < paraSize; i++){
			w[i] = Math.random() * C;
		}
		for (int i = 0; i < paraSize; i++){
			int d = i / Environment.dataCateNum;
			int y = i % Environment.dataCateNum;
			int yd = ((Setting1) v4).getLabel(d);
			if (y == yd) l[i] = 0;
			else l[i] = deltaL; 
		}
	}

	/*
	 * test ( from 100 to 10000 - 1 )
	 * regard p_0(z = i|Z) = number[i] / trainSize;
	 */
	private void test(Data v4, int setSize, int trainSize) {
		double pComponent[][] = new double[modelNum][Environment.maxComponent]; // record component weight for each, and for each model
		Vector8 etaPost[][] = new Vector8[modelNum][Environment.maxComponent]; 
		Vector4 gammaPost[][] = new Vector4[modelNum][Environment.maxComponent];
		double predictValue[] = new double[modelNum]; // record predicting value of one data from each model
		
		Vector8 f_dis = new Vector8();
		// sample models
		int [] componentNum = new int[modelNum];
		int [] numberData = new int[Environment.maxComponent];// in each component
	
		// read w -- training result
//		readW(new String("res/paraW(delta10.0k=5000Times200).txt"));  // delta 20 ?
//		readW(new String("res/paraW(delta10.0k=200Times200).txt"));  // delta 15 renewed sampling
		readW(new String("res/paraW(delta10.0k=1000Times200).txt")); // alg 3_1 10
		// construct markov chain
		MCMC t = new MCMC(v4, trainSize, w, alphaDP);
		t.go3();
		
		for (int i = 0; i < modelNum; i++){
			for (int j = 0; j < 50; j++)
				t.oneSample3();
			t.inferenceGamma();
			
			componentNum[i] = t.getCateNumer();  // Note: from 1 to componentNum[1], not componentNum[] - 1
			numberData = t.getNumberEachCate();
			pComponent[i] = new double[componentNum[i]+1];
			gammaPost[i] = new Vector4[componentNum[i]+1];
			etaPost[i] = new Vector8[componentNum[i]+1];
			// Probability of component
			for (int j = 1; j < componentNum[i]+1; j ++){
				pComponent[i][j] = (numberData[j] * 1.0) / (trainSize*1.0);
				etaPost[i][j] = new Vector8();
				gammaPost[i][j] = new Vector4();
				//System.out.print(pComponent[i][j] + " ");
			}
			//System.out.println();
			// eta estimation for each component
			for (int j = 1; j < componentNum[i]+1; j ++){
				if (pComponent[i][j] == 0){ 
					etaPost[i][j].reset();
					gammaPost[i][j].reset();
				}
				else{
					etaPost[i][j] = t.getMiu(j);
					gammaPost[i][j] = t.getGamma(j);
				}
			}
		}
		
		double []prob = new double[Environment.maxComponent];
		double []value = new double[Environment.maxComponent];
		double aver = 0;
		int predict = -1;
		Vector4 da = new Vector4();
		// predict
		for (int i = trainSize; i < setSize; i++){
//		for (int i = 0; i < trainSize; i++){	
			f_dis.setValue(((Setting1) v4).deltaF_d(i, 0));
			da = ((Setting1) v4).getV4(i);
			predict = -1;
			aver = 0;
		
			int yy = ((Setting1) v4).getLabel(i);

			for (int j = 0; j < modelNum; j++){
				predictValue[j] = 0;
				// compute p(z[d] = k), k = 1,2,3,...,cateIndexMax
				double sumt = 0;
				for (int k = 1; k <= componentNum[j]; k++){
					prob[k] = pComponent[j][k] * Math.exp(-0.5 * (da.minus(gammaPost[j][k])).norm2());
					sumt += prob[k];
				}
				for (int k = 1; k <= componentNum[j]; k++){
					prob[k] = prob[k] / sumt;
				}
				for (int k = 1; k <= componentNum[j]; k++){
					value[k] = etaPost[j][k].multiply(f_dis);
				}
				// compute prediction value
				for (int k = 1; k <= componentNum[j]; k++){
					predictValue[j] += prob[k] * value[k];
//					aver += prob[k] * value[k];
				}
			}
			for (int j = 0; j < modelNum; j++){
				aver += predictValue[j];
			}
			aver = aver / (1.0 * modelNum);
			if (aver < 0) predict = 1;
			else predict = 0;
//				System.out.println("\nmodelNum = "+modelNum);
//				for (int k = 1; k <= componentNum[j]; k++){
//					System.out.println(pComponent[j][k] +" " + etaPost[j][k].multiply(f_dis));
//				}
			
			System.out.println("Average score for i = "+i+"  with label "+yy +" : " +aver+"   "+(yy == predict));
			if (predict == yy)
				score ++;
		}
		System.out.println("score is "+score);
	}
	
	/*
	 * test ( from 100 to 10000 - 1 )
	 * regard p_0(z = i|Z) = number[i] / trainSize;
	 */
	private void test2(Data v4, int setSize, int trainSize) {
		double pComponent[][] = new double[modelNum][Environment.maxComponent]; // record component weight for each, and for each model
		Vector8 etaPost[][] = new Vector8[modelNum][Environment.maxComponent]; 
		Vector4 gammaPost[][] = new Vector4[modelNum][Environment.maxComponent];
		Vector8 f_dis = new Vector8();
		
		// read w -- training result
//		readW(new String("res/paraW(delta10.0k=5000Times200).txt"));  // delta 20 ?
		readW(new String("res/paraW(delta10.0k=200Times200).txt"));  // delta 15 renewed sampling
		
		// construct markov chain
		MCMC t = new MCMC(v4, trainSize, w, alphaDP);
		t.go2();
		
		// sample models
		int [] componentNum = new int[modelNum];
		int [] numberData = new int[Environment.maxComponent];// in each component
		
		for (int i = 0; i < modelNum; i++){
			for (int j = 0; j < 50; j++)
				t.oneSample2();
			t.inferenceGamma();
//			t.go();
			componentNum[i] = t.getCateNumer();  // Note: from 1 to componentNum[1], not componentNum[] - 1
			numberData = t.getNumberEachCate();
			pComponent[i] = new double[componentNum[i]+1];
			gammaPost[i] = new Vector4[componentNum[i]+1];
			etaPost[i] = new Vector8[componentNum[i]+1];
			// Probability of component
			for (int j = 1; j < componentNum[i]+1; j ++){
				pComponent[i][j] = (numberData[j] * 1.0) / (trainSize*1.0);
				etaPost[i][j] = new Vector8();
				gammaPost[i][j] = new Vector4();
				//System.out.print(pComponent[i][j] + " ");
			}
			//System.out.println();
			// eta estimation for each component
			for (int j = 1; j < componentNum[i]+1; j ++){
				if (pComponent[i][j] == 0){ 
					etaPost[i][j].reset();
					gammaPost[i][j].reset();
				}
				else{
					etaPost[i][j] = t.getMiu(j);
					gammaPost[i][j] = t.getGamma(j);
				}
			}
		}
		
		double []prob = new double[Environment.maxComponent];
		double aver = 0;
		int predict = -1;
		Vector4 da = new Vector4();
		// predict
//		for (int i = trainSize; i < setSize; i++){
		for (int i = 0; i < trainSize; i++){	
			f_dis.setValue(((Setting1) v4).deltaF_d(i, 0));
			da = ((Setting1) v4).getV4(i);
			predict = -1;
			aver = 0;

			for (int j = 0; j < modelNum; j++){
				// compute p(z[d] = k), k = 1,2,3,...,cateIndexMax
				double max = 0;
				int index = 0;
				for (int k = 1; k <= componentNum[j]; k++){
					prob[k] = pComponent[j][k] * Math.exp(-0.5 * (da.minus(gammaPost[j][k])).norm2());
					if (prob[k] > max){
						index = k;
						max = prob[k];
					}
				}
				// compute prediction value
				aver += etaPost[j][index].multiply(f_dis);
			}
			aver = aver / (1.0 * modelNum);
			if (aver < 0) predict = 1;
			else predict = 0;
//				System.out.println("\nmodelNum = "+modelNum);
//				for (int k = 1; k <= componentNum[j]; k++){
//					System.out.println(pComponent[j][k] +" " + etaPost[j][k].multiply(f_dis));
//				}
			int yy = ((Setting1) v4).getLabel(i);
//			System.out.println("Average score for i = "+i+"  with label "+yy +" : " +aver+"   "+(yy == predict));
			
			if (predict == yy)
				score ++;
		}
		System.out.println("score is "+score);
	}
	
	private double getP_0(int j) {
		double a1 = 0, a2 = 0; // a1 for eta, a2 for Z
//		double sum = 0;
//		for (int i = 0; i < sampleNum; i++){
//			a1 += norm2(eta[z[i]],etaLength1);
//			sum += z[i];
//		}
		a1 = norm2(eta[z[j]],etaLength1);
		a1 = Math.exp(-0.5 * a1);
		a1 *= Math.pow(2*Math.PI, -0.5 * etaLength1);
		a2 = 1/(1+alphaDP) * Math.pow(alphaDP / (1+alphaDP), z[j] - 1);
		return a1 * a2;
	}

	private double norm2(double[] ds, int etaLength12) {
		double r = 0;
		for (int j = 0; j < etaLength12; j++){
			r += ds[j] * ds[j];
		}
		return r;
	}
	
	public void evaluation() {
		accuracy = score * 1.0 / Environment.testSize;
		acc.add(accuracy);
		average += accuracy;
		printResult();
	}
	
	
	/*
	 * here I use exp_F(x,y,eta,z) (in paper iSVM, discriminative function) as the likelihood function.(problem //TODO)
	 */
	private void step1(Data v4, int setSize, int trainSize) {
		int c;
		double accept,r,m1,m2;
		for (int k = 0; k < Environment.trainSize; k++){
			for (int j = 0; j < R; j++){
				c = drawC();//  category;z
				if (z[k] == c)
					continue;
				if (number[c] == 0){ // if c* is not in {c1,c2,...,cn}, draw theta from G_0, for c*
//					drawTheta(c);
					drawGamma(c);
					drawEta(c);
				}
				// compute the acceptance probability; given f(x,y),eta,z
				// that is,f(in Setting1), eta[z[i]][], and eta[c]
				m1 = ((Setting1) v4).disFunc(eta[z[k]], k);
//				System.out.println("old "+m1);
				m2 = ((Setting1) v4).disFunc(eta[c], k);
//				System.out.println("new "+m2);
//				System.out.println("c = "+c);
				if (m2 >= m1)
					accept = 1;
				else
//					accept = m2/m1;
					accept = Math.exp(m2 - m1);
				r = Math.random();
				if (r <= accept){
					updateCate(k,c);
//					printDataStateTheta();
				}
				
			}
			if (Observed < Environment.trainSize)
				Observed ++;
		}
		
	}
	
	/*
	 * draw gamma from G_0, that is , 4-dim gaussian 
	 */
	private void drawGamma(int c) {
		double r[] = new double [gammaLength1];
		for (int i = 0 ; i < gammaLength1 ; i++){
			r[i] = sampleStandardNormalUnivariate();
		}
		vectorAssign(gamma[c],gammaLength1,r);
		return;
	}

	/*
	 * draw Eta from G_0, that is , 8-dim gaussian 
	 */
	private void drawEta(int c) {
		double r[] = new double [etaLength1];
		for (int i = 0 ; i < etaLength1 ; i++){
			r[i] = sampleStandardNormalUnivariate();
		}
		vectorAssign(eta[c],etaLength1,r);
		return;
	}
	private double[] drawEta() {
		double etaa[] = new double[etaLength1];
		double r[] = new double [etaLength1];
		for (int i = 0 ; i < etaLength1 ; i++){
			r[i] = sampleStandardNormalUnivariate();
		}
		vectorAssign(etaa,etaLength1,r);
		return etaa;
	}
	
	private void vectorAssign(double[] a, int size, double[] b){
		for (int i = 0; i < size; i++){
			a[i] = b[i];
		}
		return;
	}
	

	/*
	 * draw a candidate c*, from the conditional prior by (5.4)
	 */
	public int drawC(){
		int n;
		if (cateAlive.isEmpty()){  // no category at present, draw a new one
			return drawNewC();
		}
		else{
			double p;
			double F = 0;
			double r = Math.random();
			Iterator<Integer> it = cateAlive.iterator();
			while (it.hasNext()){
				n = (Integer)it.next();             // the nth category
				p = number[n] * 1.0 / ((Observed * 1.0) + alphaDP - 1);
				F += p;
				if (r < F)
					return n;
			}
			// a new category
			return drawNewC();
		}
	}
	private int drawNewC() {
		return minN;
	}

	private void step2(Data v4, int setSize, int trainSize) {
		for (int i = 1; i <= cateIndexMax; i++){
			if (number[i]<=0)
				continue;		
//			eta[i] = updateEta(v4,i);
		}
	}
	
	/*
	 * update eta[z] according to the posterior distribution TODO
	 * use rejection sampling.
	 * ((Setting1) v4)
	 */
	private double[] updateEta(Data v4, int com) {
		double result[] = new double [etaLength1];
		// which data is related to Component com
		int[] da = new int[100];  // record this info
		int size = 0;
		for (int j = 0; j < Environment.trainSize; j ++){
//			if (number[j] <= 0) continue;
//			System.out.println("z["+j+"] = "+z[j]);
			if (z[j] == com){
				da[size] = j;
				size++;
			}
		}
		// given prior and related data, sample an eta from posterior
		result = postSample(v4,da,size);
		 
		return result;
	}

	//given index v4, size,da
	private double[] postSample(Data v4, int[] da, int size) {
		double k = Math.pow(Math.E, 15);
		double r = 0;
		double result[] = new double [etaLength1];
		boolean success = false;
		double sumF = 0;
		double max = 0;
		int a = 0;
		int time = 0;
		do{
			time++;
			sumF = 0;
			result = drawEta();	
//			q_eta = getProG_0(result);
//			r = Math.random() * k * q_eta; 
			r = Math.random() * k; // omit q_eta because q_eta = p_eta_G0
			for (int i = 0; i < size; i++){
				sumF += ((Setting1) v4).disFunc(result, da[i]);
			}
			sumF /= Environment.reduce;
//			System.out.println("sumf =  "+sumF+"  size = "+size);
			if (sumF > max ) max = sumF;
			if ( r < Math.exp(sumF)) success = true;
			a ++;
			if (a > 400) {k *= 0.5; a = 0;}
		}while(success == false);
//		System.out.println(k+" total time "+time+"  sumf =  "+sumF+"  size = "+size);
//		System.exit(0);
		return result;
	}

	// wrong!!: use c to update state[i] and all other state with the same state
	// correct: use c to update state[i] with c
	private void updateCate(int i, int c){
		int previous = z[i];
		// case 0: state[i] == 0;
		//          change state[i] to c; if (number[c]>0), else cateNumber++ ;number[c]++; if (c > cateIndexMax) cateIndexMax = c; MinN++ until MinN not used;
		//          if cateAlive contains c , else cateAlive add c;
		if (previous == 0){				// state
			z[i] = c;
			if (number[c] <= 0){		// cateNumber,number[]
				cateNumber++;
			}
			number[c]++;
			if (c > cateIndexMax)			// cateIndexMax
				cateIndexMax = c;
			if (number[minN]!=0){
				do{ // minN
					minN ++;
				}while (number[minN]!=0);
			}
			if (!cateAlive.contains(c)){  // cateAlive
				cateAlive.add(c);
			}
		}
		else {
			// case 1: state i exists, so does c
			//		    change all state with the same state[i] to be c;number[c] += number[i],number[i] = 0;cateNumber--;if (state[i] == cateIndexMax) 
			//          cateIndexMax -- until number[cateIndexMax - 1]>0; MinN: i < MinN then MinN = i;  if cateAlive contains state[i], remove it.
			// case 2: state i exists, but not c  : new category
			//			change all state with state[i] to c; number[c] = number[i],number[i] = 0;if (state[i] == cateIndexMax)cateIndexMax -- until number[cateIndexMax - 1]>0;
			//          MinN: if i < MinN then MinN = i; if cateAlive contains state[i], remove it. cateAlive adds c;
			
			if (number[c] > 0) {// case 1
				if (number[previous]==1){
					cateNumber--;
					if (previous < minN) minN = previous;
					if (cateIndexMax == previous){
						do {
							cateIndexMax--;
						}while(number[cateIndexMax] == 0);
					}
				}
			}
			else{
				cateNumber++;
				if (number[previous]==1){
					cateNumber--;
					if (previous < c){
						minN = previous;
					}
					else{// previous > c. at this time c = minN for sure.
						do { // minN
							minN ++;
						}while( number[minN]!=0 && minN != previous);
					}	
				}
			}
			z[i] = c;
			number[c] ++;
			number[previous] --;
			while (number[minN]!=0){
				minN++;
			}
			if (c > cateIndexMax){ 
				cateIndexMax = c;
			}
			if (number[previous] == 0 && cateAlive.contains(previous)){
				int index = cateAlive.indexOf(previous);
				cateAlive.remove(index);
			}
			if (!cateAlive.contains(c)){
				cateAlive.add(c);
			}
		}
	}
	
	private void printResult(){
		System.out.println("Test Result\n---------------------------");
		System.out.print("    cateIndexMax: "+cateIndexMax+"\n    cateNumber:   "+cateNumber+"\n    MinN:         "+minN+"\n");
//		for (int j = 0; j <= cateIndexMax; j++){
//			if (number[j] <= 0)
//				continue;
//			System.out.print("    Category "+j+" ("+number[j]+"), eta = \n        ");
//			for (int i = 0; i < 8; i++){
//				System.out.print(eta[j][i]+"----");
//			}
//			System.out.println();
//		}
		
		System.out.println("    Final Score is "+score+" out of "+Environment.testSize+"\n\t\t\t ("+accuracy+")");
//		System.out.println("change time is "+change);
		System.out.println();System.out.println();
	}

	public void report() {
		average /= Environment.dataSetNum;
		System.out.println();System.out.println();
		System.out.print("Accuracy : " + average);
		
		Iterator<Double> it = acc.iterator();
		double var = 0;
		while (it.hasNext()){
			var += Math.pow((Double)it.next()-average, 2);             // the nth category
		}
		var /= Environment.dataSetNum;
		System.out.print("\nVariance : " + var+"\nSqrt(var): "+Math.sqrt(var));
		System.out.println();System.out.println();
		System.out.println("DataLable:   "+data0+" "+data1);
		System.out.println("predictLable:"+predic0+" "+predic1);
		System.out.print(data0* 1.0 / data1);
		System.out.println("/"+predic0* 1.0 / predic1);
		System.out.println();System.out.println();
	}
	
	public double sampleStandardNormalUnivariate(){ // return a sample from standard normal distribution
		double r1 = 1,r2 = 1;
		r1 = Math.random();r2 = Math.random();
		return Math.sqrt(-2 * Math.log(r1)) * Math.sin(2 * Math.PI * r2);
	}
	
	public double sampleNormalUnivariate(double mean, double var ){ // variance = var = sigma * sigma 
		double a = sampleStandardNormalUnivariate();
		return Math.sqrt(var) * a + mean;
	}
	private void trainEndLog() {
		Log good = new Log("res/paraW(delta"+stoppingCriterion+"k="+maxIteration+"Times"+Environment.sampleTimes+").txt");
		good.outln("we success break from \"while\" and now record w[]");
		for (int i = 0; i < paraSize; i ++){
			good.outln("w["+i+"] = "+w[i]);
		}
		good.close();
	}
}