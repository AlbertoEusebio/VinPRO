#include "vinpro_arduino/scissors_controller.hpp"

#include <cstring>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <pluginlib/class_list_macros.hpp>

namespace vinpro_arduino
{

// ── Lifecycle ────────────────────────────────────────────────────────────────

hardware_interface::CallbackReturn ScissorsController::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  serial_port_ = info.hardware_parameters.at("serial_port");   // e.g. /dev/ttyACM0
  baud_rate_   = std::stoi(info.hardware_parameters.at("baud_rate"));
  timeout_s_   = std::stod(info.hardware_parameters.at("timeout_s"));

  RCLCPP_INFO(rclcpp::get_logger("ScissorsController"),
    "Configured: port=%s, baud=%d, timeout=%.1fs",
    serial_port_.c_str(), baud_rate_, timeout_s_);

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn ScissorsController::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  hw_position_ = 0.0;
  hw_velocity_ = 0.0;
  hw_command_  = 0.0;
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn ScissorsController::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  if (!open_serial()) {
    return hardware_interface::CallbackReturn::ERROR;
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn ScissorsController::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  close_serial();
  return hardware_interface::CallbackReturn::SUCCESS;
}

// ── Interface exports ─────────────────────────────────────────────────────────

std::vector<hardware_interface::StateInterface>
ScissorsController::export_state_interfaces()
{
  return {
    hardware_interface::StateInterface(
      "scissors_joint", hardware_interface::HW_IF_POSITION, &hw_position_),
    hardware_interface::StateInterface(
      "scissors_joint", hardware_interface::HW_IF_VELOCITY, &hw_velocity_),
  };
}

std::vector<hardware_interface::CommandInterface>
ScissorsController::export_command_interfaces()
{
  return {
    hardware_interface::CommandInterface(
      "scissors_joint", hardware_interface::HW_IF_POSITION, &hw_command_),
  };
}

// ── Read / Write ──────────────────────────────────────────────────────────────

hardware_interface::return_type ScissorsController::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // State is updated in write() after a successful command.
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type ScissorsController::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // Only act when the command differs from the current state.
  constexpr double kHysteresis = 0.5;
  bool want_closed = (hw_command_ > kHysteresis);
  bool is_closed   = (hw_position_ > kHysteresis);

  if (want_closed == is_closed) {
    return hardware_interface::return_type::OK;
  }

  char cmd = want_closed ? 't' : 'o';
  if (!send_command(cmd)) {
    RCLCPP_ERROR(rclcpp::get_logger("ScissorsController"),
      "Arduino did not acknowledge command '%c' — aborting", cmd);
    return hardware_interface::return_type::ERROR;
  }

  hw_position_ = want_closed ? 1.0 : 0.0;
  return hardware_interface::return_type::OK;
}

// ── Serial helpers ─────────────────────────────────────────────────────────────

bool ScissorsController::open_serial()
{
  fd_ = ::open(serial_port_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
  if (fd_ < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("ScissorsController"),
      "Failed to open serial port %s: %s",
      serial_port_.c_str(), std::strerror(errno));
    return false;
  }

  // Configure raw 8N1 mode
  struct termios tty{};
  ::tcgetattr(fd_, &tty);

  speed_t speed = (baud_rate_ == 115200) ? B115200 : B9600;
  ::cfsetispeed(&tty, speed);
  ::cfsetospeed(&tty, speed);

  tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);
  tty.c_lflag = 0;   // raw input
  tty.c_oflag = 0;   // raw output
  tty.c_iflag &= ~(IXON | IXOFF | IXANY | ICRNL);

  // Blocking read with timeout (tenths of seconds)
  tty.c_cc[VMIN]  = 0;
  tty.c_cc[VTIME] = static_cast<cc_t>(timeout_s_ * 10);

  ::tcsetattr(fd_, TCSANOW, &tty);
  RCLCPP_INFO(rclcpp::get_logger("ScissorsController"),
    "Serial port %s opened at %d baud", serial_port_.c_str(), baud_rate_);
  return true;
}

void ScissorsController::close_serial()
{
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

bool ScissorsController::send_command(char cmd)
{
  if (fd_ < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("ScissorsController"),
      "Serial port not open");
    return false;
  }

  if (::write(fd_, &cmd, 1) != 1) {
    RCLCPP_ERROR(rclcpp::get_logger("ScissorsController"),
      "Failed to write command '%c': %s", cmd, std::strerror(errno));
    return false;
  }

  // Wait for 'A' acknowledgement within timeout
  char ack = '\0';
  ssize_t n = ::read(fd_, &ack, 1);
  if (n != 1 || ack != 'A') {
    RCLCPP_ERROR(rclcpp::get_logger("ScissorsController"),
      "No acknowledgement for '%c' (got '%c', n=%zd)", cmd, ack, n);
    return false;
  }
  return true;
}

}  // namespace vinpro_arduino

PLUGINLIB_EXPORT_CLASS(
  vinpro_arduino::ScissorsController,
  hardware_interface::SystemInterface)
