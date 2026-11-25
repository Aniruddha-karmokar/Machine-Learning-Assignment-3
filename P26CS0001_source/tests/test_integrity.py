"""Scientific checks that test contracts rather than copy implementation."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import log_loss
from common import *
from models import *

class Integrity(unittest.TestCase):
    def test_metric_reference(self):
        rng=np.random.default_rng(2); logits=rng.normal(size=(80,10)); y=rng.integers(0,10,80)
        m=metrics(logits,y)
        self.assertAlmostEqual(m['nll'],log_loss(y,softmax(logits,axis=1),labels=np.arange(10)))
        self.assertEqual(np.array(m['confusion']).sum(),80)
        self.assertTrue(0<=m['brier']<=2 and 0<=m['ece']<=1)
    def test_parameter_fairness_and_shapes(self):
        a,b=MLP(),CNN(); self.assertLess(abs(count(a)-count(b))/count(a),.01)
        self.assertEqual(tuple(b(torch.zeros(2,3,32,32)).shape),(2,10))
        for d in [16,64,256]:
            m=Autoencoder(d); x=torch.rand(2,3,32,32); z=m(x)
            self.assertEqual(z.shape,x.shape); self.assertTrue(z.min()>=0 and z.max()<=1)
            self.assertEqual(m.features(x).shape,(2,d))
    def test_saved_split_and_corruption(self):
        p=ROOT/'artifacts/splits'; z=np.load(p/'split.npz')
        self.assertEqual(len(z['train']),45000); self.assertEqual(len(z['val']),5000)
        self.assertEqual(len(np.unique(np.r_[z['train'],z['val']])),50000)
        c=np.load(p/'memorisation.npz'); self.assertTrue(np.isin(c['indices'],z['train']).all())
        self.assertTrue((c['target'][c['mask']]!=c['original'][c['mask']]).all())
        self.assertTrue((c['target'][~c['mask']]==c['original'][~c['mask']]).all())
        self.assertEqual(c['mask'].sum(),2000)
        self.assertTrue(np.isin(np.load(p/'svm_subset.npy'),z['train']).all())
    def test_transform_inversion_and_transport(self):
        seed_all(17); img=Images(); x=torch.randint(0,256,(3,3,32,32),device=DEVICE,dtype=torch.uint8)
        a=img.pixels(x); b=img.pixels(x,'permuted'); inv=torch.argsort(img.perm)
        self.assertTrue(torch.equal(a,b.flatten(2)[:,:,inv].reshape_as(a)))
        self.assertTrue(torch.equal(torch.sort(a.flatten(2))[0],torch.sort(b.flatten(2))[0]))
        m=MLP().to(DEVICE); z=m(img(x)); p=(img.perm[None,:]+torch.arange(3,device=DEVICE)[:,None]*1024).flatten()
        with torch.no_grad(): m.layers[1].weight.copy_(m.layers[1].weight[:,p])
        torch.testing.assert_close(z,m(img(x,'permuted')),atol=1e-6,rtol=1e-5)
        self.assertTrue(torch.equal(corrupt(a,591),corrupt(a,591)))
        self.assertFalse(torch.equal(corrupt(a,591),corrupt(a,592)))
    def test_ovo_sign_by_votes(self):
        from sklearn.svm import SVC
        rng=np.random.default_rng(8); x=np.r_[rng.normal(-2,.3,(10,2)),rng.normal(0,.3,(10,2)),rng.normal(2,.3,(10,2))]; y=np.repeat([0,1,2],10)
        m=SVC(kernel='rbf',decision_function_shape='ovo').fit(x,y); d=m.decision_function(x); votes=np.zeros((30,3))
        for col,(i,j) in enumerate([(0,1),(0,2),(1,2)]):
            votes[:,i]+=(d[:,col]>0); votes[:,j]+=(d[:,col]<=0)
        np.testing.assert_array_equal(m.predict(x),votes.argmax(1))

if __name__=='__main__': unittest.main()
