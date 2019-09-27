from Box2D import b2ContactListener
from time import sleep

class myContactListener(b2ContactListener):

    def __init__(self):
        b2ContactListener.__init__(self)

    def BeginContact(self, contact):
    	pass

    def EndContact(self, contact):
    	contact.fixtureA.body.userData = contact.fixtureB.body
    	contact.fixtureB.body.userData = contact.fixtureA.body
    def PreSolve(self, contact, oldManifold):
        pass
    def PostSolve(self, contact, impulse):
        pass
		

		