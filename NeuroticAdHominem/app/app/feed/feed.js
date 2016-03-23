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

    socket.on("eval::response", function(resp){
        $scope.evalResponse = resp;
    });

    $scope.evalString = function(){
        socket.emit("eval", $scope.stringToEval);
    }
}]);
