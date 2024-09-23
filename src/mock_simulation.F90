program main

use mpi
use smartredis_client, only : client_type
use smartredis_dataset, only : dataset_type

implicit none

integer, parameter, dimension(4) :: IDS = (/ 19909, 20450, 20966, 21480 /)
real, parameter :: LOOP_WAIT = 1
integer, parameter :: NUM_COLUMNS = 9
integer, parameter :: BLOCK_SIZE = 16
integer, parameter :: NUM_ELEMENTS = NUM_COLUMNS*BLOCK_SIZE
character(len=64), parameter :: DS_LIST = "simulation_data"

integer :: ierr, rank

character(len=256) :: line
character(len=256) :: fname

integer :: loop_index

integer :: i
integer :: ios, unit

real, dimension(BLOCK_SIZE) :: x, y, z, avg_af, avg_relV, H_index, var_af, drift_flux, avg_dPdy

type(client_type) :: client
integer :: sr_return

! Initialize MPI
rank = 0
call MPI_Init(ierr)
call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

! Open the correct data file for this
write(fname, "(A12,I5,A16)") "data/fml_plt", IDS(rank+1), "_lev0_fs0032.csv"
open(unit=unit, file=fname, action='read', iostat=ios)
read(unit, '(A)', iostat=ios) line ! Skip the header file

! Initialize the SmartRedis client
sr_return = client%initialize()
if (sr_return /= 0) call MPI_Abort(MPI_COMM_WORLD, -1, ierr)

do while (.true.)

    loop_index = loop_index + 1
    x(:) = 0.
    y(:) = 0.
    z(:) = 0.
    avg_af(:) = 0.
    avg_relV(:) = 0.
    H_index(:) = 0.
    var_af(:) = 0.
    drift_flux(:) = 0.
    avg_dPdy(:) = 0.

    do i = 1, BLOCK_SIZE
        read(unit, '(A)', iostat=ios) line
        if (ios /= 0) exit

        read(line, *) &
            x(i), y(i), z(i), avg_af(i), avg_relV(i), H_index(i), var_af(i), drift_flux(i), avg_dPdy(i)
    enddo
    if (ios/=0) exit

    call send_data(client, x, y, z, avg_af, avg_relV, H_index, var_af, drift_flux, avg_dPdy, &
                   loop_index, IDS(rank+1), BLOCK_SIZE, DS_LIST)
    call print_data(x, y, z, avg_af, avg_relV, H_index, var_af, drift_flux, avg_dPdy, BLOCK_SIZE)
    call sleep(LOOP_WAIT)
    ! call send_data()


enddo


call MPI_finalize(ierr)

contains

!> Sleep for a specified amount of time
subroutine sleep(delay)
    real, intent(in) :: delay !< How long to sleep (s)

    double precision :: start_time

    start_time = MPI_WTIME()
    do while (MPI_WTIME() - start_time < delay)
    enddo

end subroutine sleep

!> Print the data array
subroutine print_data(x, y, z, avg_af, avg_relV, H_index, var_af, drift_flux, avg_dPdy, block_size)
    real, dimension(:), intent(in) :: x
    real, dimension(:), intent(in) :: y
    real, dimension(:), intent(in) :: z
    real, dimension(:), intent(in) :: avg_af
    real, dimension(:), intent(in) :: avg_relV
    real, dimension(:), intent(in) :: H_index
    real, dimension(:), intent(in) :: var_af
    real, dimension(:), intent(in) :: drift_flux
    real, dimension(:), intent(in) :: avg_dPdy
    integer,            intent(in) :: block_size

    integer :: i

    do i = 1, block_size
        write(*, "(9ES16.8)") x(i), y(i), z(i), avg_af(i), avg_relV(i), H_index(i), var_af(i), drift_flux(i), avg_dPdy(i)
    enddo
    write(*,*) "---"

end subroutine print_data

!> Send the data array via SmartRedis
subroutine send_data(client, x, y, z, avg_af, avg_relV, H_index, var_af, drift_flux, avg_dPdy, loop_index, id, block_size, ds_list)
    type(client_type) :: client
    real, dimension(:), intent(in) :: x
    real, dimension(:), intent(in) :: y
    real, dimension(:), intent(in) :: z
    real, dimension(:), intent(in) :: avg_af
    real, dimension(:), intent(in) :: avg_relV
    real, dimension(:), intent(in) :: H_index
    real, dimension(:), intent(in) :: var_af
    real, dimension(:), intent(in) :: drift_flux
    real, dimension(:), intent(in) :: avg_dPdy
    integer,            intent(in) :: loop_index
    integer,            intent(in) :: id
    integer,            intent(in) :: block_size
    character(len=*),   intent(in) :: ds_list

    character(len=128) :: ds_name
    type(dataset_type) :: ds
    integer :: sr_return

    write(ds_name, "(A15, I0.5, A4, I5)") "data_iteration_", loop_index, "_ID_", id

    sr_return = ds%initialize(ds_name)
    sr_return = ds%add_tensor("x", x(:), (/block_size/))
    sr_return = ds%add_tensor("y", y(:), (/block_size/))
    sr_return = ds%add_tensor("z", z(:), (/block_size/))
    sr_return = ds%add_tensor("avg_af", avg_af(:), (/block_size/))
    sr_return = ds%add_tensor("avg_relV", avg_relV(:), (/block_size/))
    sr_return = ds%add_tensor("H_index", H_index(:), (/block_size/))
    sr_return = ds%add_tensor("var_af", var_af(:), (/block_size/))
    sr_return = ds%add_tensor("drift_flux", drift_flux(:), (/block_size/))
    sr_return = ds%add_tensor("avg_dPdy", avg_dPdy(:), (/block_size/))

    sr_return = client%put_dataset(ds)
if (sr_return /= 0) call MPI_Abort(MPI_COMM_WORLD, -1, ierr)
    sr_return = client%append_to_list(ds_list, ds)
if (sr_return /= 0) call MPI_Abort(MPI_COMM_WORLD, -1, ierr)
    ! do i = 1, block_size
    !     write(*, "(9ES12.8)") x(i), y(i), z(i), avg_af(i), avg_relV(i), H_index(i), var_af(i), drift_flux(i), avg_dPdy(i)
    ! enddo

end subroutine send_data


end program main
