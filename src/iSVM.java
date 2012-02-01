import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/*
 * model: infinite SVM
 * first do the discriminative model part.(generative part later)
 */
public class iSVM {
	private static final double alphaDP = 1.0;
	private final int etaLength1 = 8; // for data setting 1
	private final int R = 5; // repeating times in Algorithm 5
	private final int Times = 100; // TODO
	private int z[];
	private double eta[][];
	private int prediction[];
	private int score = 0;
	private double accuracy = 0;
	private double average = 0;
	//test
	private int data0,data1,predic0,predic1;
	public int changeTimes = 0;
	private int change = 0;
	
	
	//TODO
	
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
		z = new int[Environment.dataSetSize];
		eta = new double[Environment.maxComponent][];
		for (int i = 0; i < Environment.maxComponent; i++){
			eta[i] = new double[etaLength1];
		}
		number = new int[Environment.maxComponent+1];
		for (int i = 0; i <= Environment.maxComponent; i++){
			number[i] = 0;
		}
		prediction = new int[Environment.dataSetSize];
	}

	public void go(Vector4 v4, int setSize, int trainSize) {
		init();
		train(v4,setSize,trainSize);
		test2(v4,setSize,trainSize);
	}
	
	private void train(Vector4 v4, int setSize, int trainSize) {
		for (int i = 0; i < Times; i ++){
			int test1 = number[1];
			step1(v4,setSize,trainSize);
			step2(v4,setSize,trainSize);
			int test2 = number[1];
			if (test1 != 0 && test2 == 0){
				changeTimes ++;
				change ++;
			}
		}
		
		// without learning, totally random
		/*
		for (int i = 0; i < 4; i++){
			drawEta(i);
		}
		number[1] = 33;
		number[2] = 33;
		number[3] = 34;
		cateIndexMax = 3;
		*/
	}

	private void init() {
		for (int i = 0; i <= Environment.maxComponent; i++){
			number[i] = 0;
		}
		cateNumber = 0;
		cateIndexMax = 0;
		minN = 1;
		Observed = 0;
		cateAlive.clear();
		for (int i = 0; i < Environment.trainSize; i++){
			z[i] = 0;
		}
		
		change++;//test
	}

	/*
	 * test ( from 100 to 10000 - 1 )
	 * regard p(z = i) = number[i] / trainSize;
	 */
	private void test(Vector4 v4, int setSize, int trainSize) {
		double disF0,disF1;score = 0;
		for (int i = trainSize; i < Environment.dataSetSize; i++){
			disF0 = 0;
			disF1 = 0;
			for (int j = 0; j <= cateIndexMax; j++){
				if (number[j] <= 0)
					continue;
				disF0 += v4.disFunc(eta[j],i,0) * number[j] / (1.0 * trainSize);
				disF1 += v4.disFunc(eta[j],i,1) * number[j] / (1.0 * trainSize);
			}
			if (disF0 > disF1)	prediction[i] = 0;
			else prediction[i] = 1;
			
			//test TODO
			if (prediction[i] == 0) predic0++;
			else predic1++;
			if (v4.getLable(i) == 0) data0++;
			else data1++;
			
			score += v4.lableTest(prediction[i], i);
			
		}
	}
	
	/*
	 * test ( from 100 to 10000 - 1 )
	 * regard p(z = i) = 1 / cateNumber;
	 */
	private void test2(Vector4 v4, int setSize, int trainSize) {
		double disF0,disF1;score = 0;
		for (int i = trainSize; i < Environment.dataSetSize; i++){
			disF0 = 0;
			disF1 = 0;
			for (int j = 0; j <= cateIndexMax; j++){
				if (number[j] <= 0)
					continue;
				disF0 += v4.disFunc(eta[j],i,0) * 1 / (1.0 * cateNumber);
				disF1 += v4.disFunc(eta[j],i,1) * 1 / (1.0 * cateNumber);
//				System.out.println("F =: (0)"+v4.disFunc(eta[j],i,0) * 1 / (1.0 * cateNumber));
//				System.out.println("F =: (1)"+v4.disFunc(eta[j],i,0) * 1 / (1.0 * cateNumber));
			}
			if (disF0 > disF1)	prediction[i] = 0;
			else prediction[i] = 1;
			
			//test TODO
			if (prediction[i] == 0) predic0++;
			else predic1++;
			if (v4.getLable(i) == 0) data0++;
			else data1++;
			
			int t = score; // test
			score += v4.lableTest(prediction[i], i);
//			if (t == score){ // wrong
//				System.out.print("--------------\nWrong\n   :\n");
//				v4.printV(i);
//				System.out.println(cateNumber+" component");
//				for (int j = 0; j <= cateIndexMax; j++){
//					if (number[j] <= 0)
//						continue;
//					System.out.println("com "+j+" :"+v4.disFunc(eta[j],i,prediction[i]));
//				}
//			}
//			else{
//				System.out.print("--------------\nCorrect\n   :\n");
//				v4.printV(i);
//				for (int j = 0; j <= cateIndexMax; j++){
//					if (number[j] <= 0)
//						continue;
//					System.out.println("com "+j+" :"+v4.disFunc(eta[j],i,prediction[i]));
//				}
//			}
			
		}
	}
	
	public void evaluation() {
		accuracy = score * 1.0 / Environment.testSize;
		average += accuracy;
		printResult();
	}
	
	
	/*
	 * here I use exp_F(x,y,eta,z) (in paper iSVM, discriminative function) as the likelihood function.(problem //TODO)
	 */
	private void step1(Vector4 v4, int setSize, int trainSize) {
		int c;
		double accept,r,m1,m2;
		for (int k = 0; k < Environment.trainSize; k++){
			for (int j = 0; j < R; j++){
				c = drawC();//  category;z
				if (z[k] == c)
					continue;
				if (number[c] == 0){ // if c* is not in {c1,c2,...,cn}, draw theta from G_0, for c*
//					drawTheta(c);
					drawEta(c);
				}
				// compute the acceptance probability; given f(x,y),eta,z
				// that is,f(in vector4), eta[z[i]][], and eta[c]
				m1 = v4.disFunc(eta[z[k]], k);
				m2 = v4.disFunc(eta[c], k);
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
	 * draw Eta from G_0, that is , 8-dim gaussian 
	 */
	private void drawEta(int c) {
		double r[] = new double [8];
		for (int i = 0 ; i < 8 ; i++){
			r[i] = sampleStandardNormalUnivariate();
		}
		vectorAssign(eta[c],etaLength1,r);
		return;
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

	private void step2(Vector4 v4, int setSize, int trainSize) {
//		for (int i = 1; i <= cateIndexMax; i++){
//			if (number[i]<=0)
//				continue;		
//			eta[i] = updateEta(i);
//		}
	}
	
	private double[] updateEta(int i) {
		// TODO Auto-generated method stub
		return null;
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
		for (int j = 0; j <= cateIndexMax; j++){
			if (number[j] <= 0)
				continue;
			System.out.print("    Category "+j+" ("+number[j]+"), eta = \n        ");
			for (int i = 0; i < 8; i++){
				System.out.print(eta[j][i]+"----");
			}
			System.out.println();
		}
		
		System.out.println("    Final Score is "+score+" out of "+Environment.testSize+" ("+accuracy+")");
//		System.out.println("change time is "+change);
		System.out.println();System.out.println();
	}

	public void report() {
		average /= Environment.dataSetNum;
		System.out.println();System.out.println();
		System.out.print("Accuracy : " + average);
		System.out.println();System.out.println();
		System.out.println("DataLable:   "+data0+" "+data1);
		System.out.println("predictLable:"+predic0+" "+predic1);
		System.out.print(data0* 1.0 / data1);
		System.out.println("/"+predic0* 1.0 / predic1);
		System.out.println();System.out.println();
	}
	
	public double sampleStandardNormalUnivariate(){ // return a sample from standard normal distribution
		double r1 = 1,r2 = 1;
		double tmp = r1*r1 + r2*r2;
		while (tmp > 1){
			r1 = 2 * Math.random() - 1;
			r2 = 2 * Math.random() - 1;
			tmp = r1*r1 + r2*r2;
		}
		return r1 * Math.sqrt(-2 * Math.log(Math.abs(r1)) / tmp);
	}
	
	public double sampleNormalUnivariate(double mean, double var ){ // variance = var = sigma * sigma 
		double a = sampleStandardNormalUnivariate();
		return Math.sqrt(var) * a + mean;
	}


}