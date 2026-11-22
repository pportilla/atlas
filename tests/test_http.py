import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from engine.app import AtlasServer
from engine.zoo_search import zoo_search
from tests.http_support import QuietHandler


class WebAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http=AtlasServer(('127.0.0.1',0),QuietHandler)
        cls.thread=threading.Thread(target=cls.http.serve_forever,daemon=True)
        cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.http.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown();cls.http.server_close();cls.thread.join()

    def post(self, obj):
        return urlopen(Request(self.url+'/api/resolve',data=json.dumps(obj).encode(),
                               headers={'Content-Type':'application/json'}),timeout=10)

    def test_browser_assets_and_resolution(self):
        with urlopen(self.url, timeout=5) as resp:
            self.assertIn('Resolution Atlas',resp.read().decode())
        with self.post({'polynomial':'y^2-x^3','variables':['x','y'],'mode':'intrinsic','compute':'cpu'}) as resp:
            r=json.load(resp)
        self.assertEqual(r['execution']['device'],'CPU')
        self.assertEqual(r['dual_complex']['vertices'][0]['id'],'P1')
        self.assertEqual(len(r['charts']),2)

    def test_catalogue_search_and_lazy_record_routes(self):
        def search(query):
            request = Request(self.url + '/api/zoo/search', json.dumps({'query': query}).encode(),
                              {'Content-Type': 'application/json'})
            with urlopen(request, timeout=5) as response:
                return json.load(response)['ids']

        with urlopen(self.url + '/zoo', timeout=5) as response:
            self.assertTrue(response.url.endswith('/zoo/'))
            self.assertIn('Search by monomials', response.read().decode())
        with urlopen(self.url + '/zoo/zoo-data/index.json', timeout=5) as response:
            entries = json.load(response)['entries']
        ordered_ids = search('')
        self.assertCountEqual(ordered_ids, [e['id'] for e in entries])
        self.assertEqual(ordered_ids, zoo_search('')['ids'])
        self.assertIn('mixed-quartic', search(['1/2*u*v^3', '-v^4']))
        self.assertNotIn('mixed-quartic', search('1/2*u*v^3, v^4'))
        with self.assertRaises(HTTPError) as invalid:
            search('1/0*u')
        self.assertEqual(invalid.exception.code, 400)
        invalid.exception.close()
        with urlopen(self.url + '/zoo/zoo-data/records/mixed-quartic.json', timeout=5) as response:
            record = json.load(response)
        self.assertEqual(record['embedded']['status'], 'resolved')
        self.assertEqual(record['zeta']['topological']['status'], 'computed')
        for path in ('records/does-not-exist.json', 'records/../../index.html'):
            with self.assertRaises(HTTPError) as missing:
                urlopen(self.url + '/zoo/zoo-data/' + path, timeout=5)
            self.assertEqual(missing.exception.code, 404)
            missing.exception.close()

    def test_invalid_or_inconclusive(self):
        with self.assertRaises(HTTPError) as err:
            self.post({'polynomial':'x/0','variables':['x','y']})
        self.assertEqual(err.exception.code,400);err.exception.close()
        # Exceed the documented quick-mode expansion limit independently
        # of which geometric fast paths the native resolver supports.
        from engine.polynomial import MAX_DEGREE
        with self.post({'polynomial':f'x^{MAX_DEGREE + 1}+y', 'variables':['x','y'], 'search':'quick'}) as resp:
            result=json.load(resp)
            self.assertEqual(result['status'],'inconclusive')
            self.assertTrue(result['can_search'])
