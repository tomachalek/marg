'''
Created on 7.2.2012

@author: Tomas Machalek <tomas.machalek@uhk.cz>
'''
import os
import unittest
import mox
import mahotas
import random

from imreg.marg.marg import *
from imreg import similarity
from imreg import image


class TileTest(unittest.TestCase):

    def test_init_error(self):
        """
        Tile should raise TypeError in case of unsupported data
        """
        img = "Lorem ipsum dolor sit amet..."
        self.assertRaises(TypeError, Tile, img, (0, 0))

    def test_accept_python_list(self):
        """
        Tile should accept also standard Python lists
        """
        img = [[0, 0, 127], [3, 4, 46], [2, 12, 0]]
        tile = Tile(img, (0, 0))
        self.assertEqual(type(tile.image), np.ndarray)

    def test_init_position(self):
        pass


class TileGeneratorTest(unittest.TestCase):
    """
    """
    
    def setUp(self):
        self.img = mahotas.imread('%s/../../../test-data/lena-64.jpg' % os.path.dirname(__file__), as_grey=True)
    
    def test_init(self):
        self.assertTrue(type(self.img), np.ndarray)
        
    def test_create_tile(self):
        tile = generate_tile(self.img, (10, 10), 10)
        self.assertEqual(np.size(tile.image, 0), 21)
        self.assertEqual(np.size(tile.image, 1), 21)
        self.assertTrue(np.all(tile.image == self.img[0:21, 0:21]))


class AgentTest(unittest.TestCase):
    '''
    Tests agent functionality
    '''
    def setUp(self):
        self.mox = mox.Mox()
        self.images = (mahotas.imread('%s/../../../test-data/lena-64.jpg' % os.path.dirname(__file__), as_grey=True),
                       mahotas.imread('%s/../../../test-data/lena-64.jpg' % os.path.dirname(__file__), as_grey=True))

    def tearDown(self):
        self.mox.UnsetStubs()

    def test_agent_iteration(self):
        """
        """
        self.mox.StubOutWithMock(random, 'randint')
        # generate agent's position
        random.randint(10, 53).AndReturn(32) # 53 = img_width - tile_radius - 1
        random.randint(10, 53).AndReturn(32) # 10 = 0 + tile_radius + 1

        # generate 2 random tiles
        random.randint(10, 53).AndReturn(32)
        random.randint(10, 53).AndReturn(32)
        random.randint(10, 53).AndReturn(15)
        random.randint(10, 53).AndReturn(15)

        # try two new positions for the agent
        random.randint(-32, 32).AndReturn(5) # let's try x = agent_x + 5
        random.randint(-32, 32).AndReturn(5) # let's try y = agent_y + 5
        random.randint(-32, 32).AndReturn(10) # let's try x = agent_x + 10
        random.randint(-32, 32).AndReturn(10) # let's try y = agent_y + 10

        self.mox.ReplayAll()
        agent_pool = AgentPool(self.images, num_agents=1, num_incom_segments=1,
            segment_radius=10, None, False)
        agent = agent_pool.agents[0]
        agent.position = agent.tile.position
        agent.iterate()

        tile_x1 = generate_tile(self.images[0], (37, 37), 10)
        ans = similarity.sad(tile_x1.image, agent.tile.image)
        self.assertAlmostEqual(agent.current_value, ans, 4)
        self.mox.VerifyAll()


class AgentPoolTest(unittest.TestCase):
   
    
    def setUp(self):
        """
        """
        self.images = (mahotas.imread('%s/../../../test-data/lena-64.jpg' % os.path.dirname(__file__), as_grey=True),
                       mahotas.imread('%s/../../../test-data/lena-64.jpg' % os.path.dirname(__file__), as_grey=True))


    def test_init(self):
        agent_pool = AgentPool(self.images, num_agents=2, num_incom_segments=2, segment_radius=10)
        self.assertEqual(len(agent_pool.agents), 2)
        self.assertEqual(len(agent_pool.tiles_per))


class ResultProcessingTest(unittest.TestCase):
    """
    """
    def test_calc_best_result(self):
        a = ((125, 127), (125, 125), (124, 126), (130, 137), (157, 201), (132, 129), (125, 200), (125, 128))
        best_coords = calc_best_result(a, threshold=0.70710678119)
        print(best_coords)