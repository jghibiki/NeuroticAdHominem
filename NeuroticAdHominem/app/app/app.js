'use strict';

// Declare app level module which depends on views, and components
angular.module('nah', [
  'ngRoute',
  'btford.socket-io',
  'nah.feed',
  'nah.view2',
  'nah.version'
]).
config(['$routeProvider', function($routeProvider) {
  $routeProvider.otherwise({redirectTo: '/nah'});
}]).
factory('socket', function (socketFactory) {
  return socketFactory({
    ioSocket: io.connect()
  });
});
