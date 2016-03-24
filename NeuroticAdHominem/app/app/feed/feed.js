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
    $scope.feed = [];

    socket.on("eval::response", function(resp){
        $scope.evalResponse = resp;
    });

    socket.on("stream:eval", function(resp){
        if($scope.feed.length > 100){
            $scope.feed.splice(0, 1);
        }
        $scope.feed.push(resp); 
    });

    $scope.evalString = function(){
        socket.emit("eval", $scope.stringToEval);
    }
}]);
