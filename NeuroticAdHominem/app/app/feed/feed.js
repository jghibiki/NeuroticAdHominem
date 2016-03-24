'use strict';

angular.module('nah.feed', ['ngRoute'])

.config(['$routeProvider', function($routeProvider) {
  $routeProvider.when('/feed', {
    templateUrl: 'feed/feed.html',
    controller: 'FeedCtrl'
  });
}])

.controller('FeedCtrl', ["$scope", "socket", function($scope, socket) {
    $scope.stringToEval = "";
    $scope.evalResponse = "";
    $scope.cleanFeed = [];
    $scope.dirtyFeed = [];
    $scope.cleanCount = 0;
    $scope.dirtyCount = 0;
    $scope.pauseScroll = false;

    socket.on("eval::response", function(resp){
        $scope.evalResponse = resp;
    });

    socket.on("stream:eval", function(resp){
        if(!$scope.pauseScroll){

            if(resp.classification == "example"){
                if($scope.dirtyFeed.length > 20){
                    $scope.dirtyFeed.splice(0, 1);
                }
                $scope.dirtyCount = $scope.dirtyCount + 1;
                $scope.dirtyFeed.push(resp); 
            }
            else{
                if($scope.cleanFeed.length > 20){
                    $scope.cleanFeed.splice(0, 1);
                }
                $scope.cleanCount = $scope.cleanCount + 1;
                $scope.cleanFeed.push(resp); 
            }
        }
    });

    $scope.evalString = function(){
        socket.emit("eval", $scope.stringToEval);
    }
}]);
