"""Quick offline test for real user-trained model bundle."""
from datetime import date, time
import math
import unittest

from inference import (load_bundle,airport_label,build_features, predict_flight,route_entry,
                       holiday_features,scheduled_times)

class FlightRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta,cls.routes,cls.clf,cls.reg=load_bundle()

    def test_assets_and_feature_alignment(self):
        self.assertEqual(len(self.meta['features']),44)
        self.assertEqual(list(self.meta['features']),list(self.clf.feature_names_))
        self.assertEqual(list(self.meta['features']),list(self.reg.feature_names_))
        self.assertGreater(len(self.routes),7000)

    def test_full_names(self):
        self.assertIn('New York',airport_label('JFK',self.meta))
        self.assertIn('International Airport',airport_label('JFK',self.meta))

    def test_features(self):
        x,dep,arr=build_features('Delta Air Lines','ATL','MCO',date(2026,11,24),time(14,0),
                                 self.meta,self.routes)
        self.assertEqual(x.shape,(1,44))
        self.assertEqual(x.at[0,'route'],'ATL → MCO')
        self.assertEqual(x.at[0,'origin_wx_available'],0)
        self.assertEqual(dep.tzinfo.key,'America/New_York')
        self.assertTrue(arr>dep)

    def test_real_models_predict(self):
        for airline,a,b in [('Delta Air Lines','ATL','MCO'),('American Airlines','DFW','LAX'),
                            ('United Airlines','ORD','EWR')]:
            p=predict_flight(airline,a,b,date(2026,11,24),time(14,0))
            self.assertTrue(0<=p.late_probability<=1)
            self.assertTrue(math.isfinite(p.arrival_delay_minutes))
            self.assertEqual(p.estimated_arrival.tzinfo,p.typical_scheduled_arrival.tzinfo)

    def test_timezone_jfk_to_lax(self):
        start,end=scheduled_times(date(2026,11,24),time(14,0),360,'JFK','LAX',self.meta)
        self.assertEqual(start.hour,14)
        self.assertEqual(end.hour,17)
        self.assertEqual(end.tzinfo.key,'America/Los_Angeles')

    def test_unknown_route_rejected(self):
        with self.assertRaises(ValueError):
            route_entry(self.routes,'Delta Air Lines','JFK','JFK')

    def test_holiday(self):
        days,exact,near=holiday_features(date(2026,12,25))
        self.assertEqual((days,exact,near),(0,1,1))

if __name__=='__main__':unittest.main(verbosity=2)
