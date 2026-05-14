#pragma once

#include <string>
#include <vector>

#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>

namespace vinpro_arduino
{

/**
 * @brief ros2_control hardware interface for the Arduino Nano electric shear.
 *
 * Exposes a single revolute joint ("scissors_joint") with:
 *   - command interface: position  (0.0 = open, 1.0 = closed)
 *   - state  interfaces: position, velocity
 *
 * Serial protocol (8N1, configurable baud rate):
 *   Send 't'  → Arduino performs cut → acknowledges with 'A'
 *   Send 'o'  → Arduino opens shear  → acknowledges with 'A'
 *
 * write() blocks until the Arduino acknowledges or the timeout expires.
 * If no acknowledgement within scissors_timeout_s, the cut is aborted
 * and the interface returns an error state.
 */
class ScissorsController : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(ScissorsController)

  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  bool open_serial();
  void close_serial();
  bool send_command(char cmd);   // sends cmd, waits for 'A', returns success

  // Serial state
  int fd_{-1};
  std::string serial_port_;
  int baud_rate_{9600};
  double timeout_s_{5.0};

  // Joint state / command mirrors
  double hw_position_{0.0};    // current position (0.0 open, 1.0 closed)
  double hw_velocity_{0.0};
  double hw_command_{0.0};     // commanded position from MTC
};

}  // namespace vinpro_arduino
