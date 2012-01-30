/*
 * model: infinite SVM
 * first do the predictive model part.(generative part later)
 */
public class iSVM {
	private int z[];
	
	iSVM(){
		z = new int[Environment.dataSetSize];
	}

	public void go(Vector4 v4, int setSize, int trainSize) {
		train(v4,setSize,trainSize);
		test(v4,setSize,trainSize);
	}

	private void train(Vector4 v4, int setSize, int trainSize) {
		// TODO Auto-generated method stub
		
	}

	private void test(Vector4 v4, int setSize, int trainSize) {
		// TODO Auto-generated method stub
		
	}
	
	public void evaluation() {
		// TODO Auto-generated method stub
		
	}
	
}